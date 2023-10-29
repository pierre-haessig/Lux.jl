


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
function (s::SpiralClassifier)(x::AbstractArray{T, 3}, ps::NamedTuple,
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
            (loss, y_pred, st), back = pullback(compute_loss, x, y, model, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
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
((lstm_cell = (weight_i = Float32[-0.8643142 -0.4230192; -0.25249168 -0.75625455; 0.80865157 0.88429147; -0.062498126 0.07473498; -0.14664596 -0.61958987; 0.2875273 0.54737; -0.83378077 0.07670456; 0.06455627 0.05994223; -0.03822851 0.082926214; -0.66346 0.45134497; 1.1802354 -0.020945052; -0.020164935 0.0866885; -0.11140742 0.24431461; -0.85376465 0.3149622; -0.12769751 -0.27462053; 1.2206303 0.09875005; 0.67954046 -0.6276859; 0.12892817 -0.3435311; 0.4587593 -0.32500786; -0.5522043 0.5176793; 0.5845432 -0.8341737; 0.09492781 0.29479814; 0.8701975 -0.6391703; 0.9463097 0.6212789; -1.2449433 -0.082102224; 0.476144 0.6431999; 1.0552448 0.67392486; -0.4616015 -0.14952312; 0.73954797 -0.69509614; -0.3410752 0.7177804; -0.2190327 0.6131945; 0.6074828 -0.28869534], weight_h = Float32[-0.5224505 -0.0576032 0.29737806 -0.25633502 0.29839602 0.0038236256 -0.71655816 0.55040485; -0.65871245 0.21769816 -0.06025295 0.62787044 0.38113868 0.3090698 0.1320076 -0.045714144; 0.014954099 0.062526666 -0.029574469 0.56212234 -0.7417896 0.31763446 -0.6371022 -0.14653784; 0.036666833 -0.26669538 0.88880813 -0.092583366 0.91303205 0.09458136 0.2779979 0.9095365; -0.36562127 0.47051138 0.7787498 0.3306378 0.4029234 0.63320917 -0.33016145 -0.07681692; -0.05497973 -0.3369931 0.3319301 -0.09737438 -0.28123617 -0.21983589 -0.4887529 0.05327449; -0.791023 0.37369123 0.68979347 0.52042353 -0.8106757 0.6360248 0.02136018 -0.6185417; 0.68020207 0.29883313 1.0694383 -1.305142 0.8115903 0.15507343 -0.7089587 1.1526656; -0.16177642 0.46988365 0.41657656 -0.49493882 0.46432713 0.6731153 0.18580078 0.6425297; -0.41846722 -0.18646696 -0.38311508 -0.0082618715 0.24149902 -0.0026182577 -0.7571882 0.73596746; -0.14914021 0.55087036 -0.04062107 0.5275419 -0.52061176 0.29363644 -0.2678281 -0.37039256; 0.054209597 0.88705695 0.29014665 -0.10495266 0.8299222 0.63982815 0.38627037 0.50213647; 0.64773947 0.49023467 0.37150195 -0.06904411 0.92266315 0.15753211 -0.71932423 0.44362637; -0.09286123 0.44428748 0.086946815 0.09229562 0.61076045 -0.016616534 -0.1411396 -0.39349714; -0.8377975 0.32623538 0.08550127 -0.39624205 -0.26423922 0.79844195 -0.37202683 0.42538935; -0.6479787 0.78767323 0.42202258 1.0784556 -0.08449814 0.8130385 -0.10009841 0.76814425; 0.14954664 -0.63975775 -0.050380543 -0.44158986 -0.3490778 -0.21517187 -0.1641899 0.17281835; -0.44133344 0.23044117 0.19594747 0.7175324 0.39366594 -0.35858878 -0.36183545 0.66545653; -0.554676 0.7060361 0.0934036 -0.42992964 -0.3182899 0.7412871 0.050171208 0.603636; -0.77592885 0.339763 -0.16858603 0.5045095 -0.69759405 -0.13485916 -0.20051809 -0.19581981; -0.26754585 -0.6645032 0.39855507 -0.64234334 -0.16842753 0.26640376 0.24894369 -0.1180424; -0.72117037 -0.43179575 0.5332965 -0.48466003 -0.13196623 0.031720273 -0.44162062 0.22563769; 0.48553768 -0.18846788 -0.66110355 0.2562694 0.34346288 0.07353066 -0.52244943 -0.41770628; -0.4134488 0.3557269 0.22096607 0.37225437 -0.30600354 -0.19074272 -0.74871504 0.066181; -0.65586776 0.54620355 0.26107514 -0.44104013 0.5997729 0.428623 -0.93780935 0.87197757; -0.6014267 0.2625922 0.07142934 0.42702886 -0.2712419 0.5580941 -0.41396412 -0.30218104; -0.81896913 -0.27285227 0.46087036 -0.278301 0.25009593 0.012733342 -0.8919746 0.7581592; 0.25609255 0.2698127 1.0173239 -0.89724374 0.2479259 0.17694269 -1.1244848 0.49443546; -0.3880577 0.2761264 0.764752 -0.42167348 0.4520062 0.8859315 -1.1940877 0.72028035; 0.45921674 0.37916583 -0.6991317 0.62558806 -1.2361019 -0.38960883 -0.14209563 -0.06741414; -0.43737593 0.5177861 -0.26554355 0.7189196 -0.82021546 -0.14478806 0.08515762 -0.08153265; -0.5614535 0.74240196 0.680466 -0.5169371 0.6545664 0.047632974 -0.48334616 0.6496626], bias = Float32[0.28977975; 0.2751811; 0.14607191; 0.32598802; 0.34491393; 0.12034107; 0.0121037075; 0.971812; 0.3805997; 0.15648426; 0.008570847; 0.35888618; 0.4330961; 0.071013786; -0.014111367; 0.3210624; 0.8448936; 1.1311157; 1.1574483; 0.7685998; 0.68883705; 1.2352933; 0.64259696; 0.9169911; 0.47300664; 0.06458586; 0.3099212; 0.62286216; 0.63956714; 0.007364438; -0.117798366; 0.8743395;;]), classifier = (weight = Float32[-1.4296229 0.766024 1.2360307 1.259734 -0.93699193 0.110560045 -0.26839417 1.2228496], bias = Float32[-0.65030485;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

