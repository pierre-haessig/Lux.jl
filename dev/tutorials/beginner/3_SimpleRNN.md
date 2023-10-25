


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


<a id='Dataset'></a>

## Dataset


We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq*len × batch*size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.


```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :,
        sequence_length, 1) for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


<a id='Creating-a-Classifier'></a>

## Creating a Classifier


We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.


We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.


To understand more about container layers, please look at [Container Layer](http://lux.csail.mit.edu/stable/manual/interface/#container-layer).


```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](../../api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](../../api/Lux/layers#Lux.Dense).


```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##292".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(x::AbstractArray{T, 3},
    ps::NamedTuple,
    st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


<a id='Defining-Accuracy,-Loss-and-Optimiser'></a>

## Defining Accuracy, Loss and Optimiser


Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.


```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
            gs = back((one(loss), nothing, nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
((lstm_cell = (weight_i = Float32[-0.8615876 -0.41898397; -0.2542846 -0.7562548; 0.8107086 0.8843693; -0.058953404 0.07770545; -0.14656094 -0.6202087; 0.28774205 0.5525511; -0.83757305 0.0708979; 0.07451122 0.05749124; -0.039095357 0.07574823; -0.6628417 0.45251998; 1.1508399 -0.017591376; -0.014007604 0.08788796; -0.10978351 0.2441878; -0.850454 0.32071966; -0.12910946 -0.2709021; 1.2187701 0.104457855; 0.68050545 -0.62708396; 0.13021514 -0.34484404; 0.4695763 -0.32456636; -0.5556276 0.51659954; 0.5834587 -0.83547324; 0.10044308 0.29458863; 0.8707284 -0.6389138; 0.9468645 0.6236792; -1.2630566 -0.09206365; 0.47936934 0.642491; 1.0546886 0.67760843; -0.43758994 -0.1530753; 0.7498594 -0.6948425; -0.34131226 0.7254335; -0.22135933 0.611187; 0.6148119 -0.29305482], weight_h = Float32[-0.51886296 -0.056105383 0.29734582 -0.2617461 0.29580468 0.009668532 -0.7062745 0.5478815; -0.6505427 0.2141296 -0.062078007 0.63122416 0.37953314 0.30982658 0.13202052 -0.04476224; 0.015695818 0.062279165 -0.03386217 0.5646751 -0.7394659 0.31587455 -0.6393599 -0.16106212; 0.034240056 -0.27101523 0.8863137 -0.09377129 0.9103601 0.09346987 0.27815312 0.90725124; -0.3718747 0.46926266 0.77721447 0.32480747 0.40092683 0.634681 -0.33639863 -0.07906001; -0.06081519 -0.33573163 0.33151582 -0.09479927 -0.28700453 -0.21961811 -0.49462986 0.046680916; -0.7893352 0.37362933 0.69260615 0.5170886 -0.81258684 0.63682854 0.02421219 -0.6217326; 0.6798758 0.30291972 1.0616283 -1.3041844 0.81202817 0.15624468 -0.7104192 1.147152; -0.16721772 0.47249538 0.4188556 -0.50178057 0.4613064 0.67814493 0.17799334 0.63849217; -0.41978955 -0.1861577 -0.3822344 -0.008453654 0.240379 -0.0007547175 -0.74302083 0.74960953; -0.15108468 0.5528441 -0.043710507 0.5282397 -0.51776046 0.29576728 -0.26939046 -0.3746077; 0.0545217 0.8843392 0.29081437 -0.10969121 0.82946974 0.6409449 0.3870869 0.5014878; 0.6482959 0.4930553 0.37486798 -0.074041426 0.9249826 0.16347627 -0.69564974 0.4450781; -0.0988684 0.4433568 0.08710927 0.08906154 0.60708004 -0.014000389 -0.13867408 -0.4008777; -0.8329488 0.32674608 0.08751224 -0.39697066 -0.26484412 0.8034237 -0.37073523 0.42131534; -0.6507529 0.7878943 0.4204264 1.0825442 -0.07197771 0.8102407 -0.09730999 0.7811915; 0.14993137 -0.6404237 -0.054957468 -0.44305095 -0.3509739 -0.21894075 -0.1642317 0.172788; -0.44041577 0.22949839 0.19706751 0.71682024 0.39144737 -0.35823813 -0.3595784 0.66773176; -0.5554768 0.70911354 0.09461837 -0.42971757 -0.314241 0.7419405 0.049132794 0.6038773; -0.77532554 0.34048367 -0.16635804 0.50351226 -0.6966041 -0.13473554 -0.20010012 -0.1950595; -0.26594105 -0.6652658 0.39708176 -0.64891785 -0.16136795 0.26509017 0.24739537 -0.12166221; -0.7272826 -0.42642412 0.5306336 -0.47690555 -0.13984269 0.02948499 -0.44311216 0.21830444; 0.4855816 -0.18325356 -0.6624829 0.26020214 0.34299564 0.072543435 -0.5222607 -0.41882747; -0.4170699 0.3564575 0.22073302 0.3745099 -0.30652457 -0.19156331 -0.7492196 0.05837285; -0.6575156 0.5492513 0.26530758 -0.45252603 0.59927356 0.43380317 -0.95107204 0.87122166; -0.60262084 0.26354998 0.067490816 0.42895856 -0.2668361 0.5622675 -0.41490325 -0.30273768; -0.8208181 -0.2738151 0.46192765 -0.26912886 0.2350461 0.013276861 -0.8915219 0.74756956; 0.26326752 0.27157295 1.0148748 -0.9111564 0.2555872 0.176588 -1.1263299 0.49292696; -0.37625805 0.27501 0.76480925 -0.42989916 0.4518838 0.88976973 -1.191062 0.7159232; 0.45011395 0.38675392 -0.69492686 0.6252706 -1.2231299 -0.37965146 -0.15321538 -0.06116896; -0.43491307 0.51892245 -0.25614053 0.7161911 -0.818982 -0.13448288 0.08496787 -0.0818499; -0.55698526 0.7446588 0.6827002 -0.5192183 0.648571 0.05074602 -0.48896664 0.6486116], bias = Float32[0.28897846; 0.27044562; 0.14069264; 0.32212552; 0.3430679; 0.12011759; 0.011660487; 0.9722623; 0.37942976; 0.1562163; 0.009439331; 0.357681; 0.43537828; 0.0685229; -0.01333343; 0.32712772; 0.8439044; 1.1298934; 1.1580019; 0.7694966; 0.6889405; 1.2380898; 0.65347934; 0.9119431; 0.47490764; 0.050385084; 0.30314714; 0.62569296; 0.63902575; 0.013880396; -0.1171379; 0.8744886;;]), classifier = (weight = Float32[-1.4325819 0.7549526 1.2363409 1.2666086 -0.9374428 0.11607117 -0.2675336 1.2128578], bias = Float32[-0.63732207;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
```


<a id='Saving-the-Model'></a>

## Saving the Model


We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model


```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model


```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

