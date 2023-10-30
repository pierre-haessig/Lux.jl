


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
((lstm_cell = (weight_i = Float32[-0.8633374 -0.4217729; -0.25322834 -0.7560895; 0.8093473 0.88312; -0.059879526 0.07513373; -0.14628652 -0.620348; 0.28829452 0.5505068; -0.8358975 0.07394779; 0.07017513 0.05872538; -0.038950548 0.0790003; -0.66270596 0.45187742; 1.1729248 -0.021641798; -0.016062725 0.086942814; -0.11034533 0.24367854; -0.85084635 0.31877378; -0.12916866 -0.27274364; 1.2198371 0.099595696; 0.6793852 -0.6267394; 0.13022925 -0.34475884; 0.47205514 -0.32519075; -0.555812 0.51737154; 0.58251536 -0.83531183; 0.10021275 0.2953199; 0.8692952 -0.6383457; 0.9436253 0.6257814; -1.2641225 -0.087487616; 0.48108974 0.64272845; 1.0555006 0.6747472; -0.44462875 -0.1540948; 0.7494413 -0.69714934; -0.34109768 0.71851414; -0.21974179 0.61255753; 0.61500067 -0.28990203], weight_h = Float32[-0.52078307 -0.056949 0.29880646 -0.26095134 0.29783228 0.005618042 -0.71566737 0.5502404; -0.6520783 0.21617731 -0.06374681 0.63254076 0.38012305 0.30994794 0.1319206 -0.04330325; 0.016862053 0.060974542 -0.03339785 0.5627927 -0.7420077 0.31506863 -0.63645524 -0.14703295; 0.035176177 -0.2700931 0.8895677 -0.09240555 0.9126504 0.09072416 0.2770664 0.9094439; -0.3677224 0.4699809 0.7792197 0.32892627 0.4026486 0.63245356 -0.33367866 -0.07673908; -0.05862117 -0.3365707 0.3321063 -0.09628666 -0.28500044 -0.21975034 -0.4927297 0.04586891; -0.7892767 0.37425613 0.6890488 0.5151634 -0.8105164 0.63584197 0.025538642 -0.61832196; 0.68348294 0.30639824 1.068952 -1.3026301 0.8083998 0.15462863 -0.7077756 1.147837; -0.16502117 0.47137782 0.4157997 -0.49646237 0.46379253 0.6747954 0.18342966 0.6413287; -0.4192582 -0.18656091 -0.3832635 -0.008633117 0.24053842 -0.0016692228 -0.7472988 0.7442556; -0.1486541 0.5515218 -0.05089811 0.5283962 -0.52119297 0.29312974 -0.26690957 -0.369253; 0.0541021 0.886289 0.2905742 -0.106558226 0.8296847 0.6342508 0.3865097 0.50158036; 0.64862853 0.49307007 0.37390748 -0.07100873 0.92387265 0.16173935 -0.7155923 0.44434327; -0.09500267 0.44263047 0.08489845 0.08878022 0.60823727 -0.018367035 -0.13839254 -0.4030711; -0.832697 0.3252365 0.08749487 -0.39855665 -0.26263717 0.80403435 -0.36922836 0.42562693; -0.64547646 0.7906698 0.41889495 1.0557349 -0.08730078 0.8151439 -0.10393683 0.7663492; 0.15056607 -0.6393416 -0.05533669 -0.44238123 -0.35023215 -0.21626793 -0.1636389 0.17362799; -0.44053975 0.2294548 0.1969646 0.71674734 0.39108133 -0.35671073 -0.36213538 0.667662; -0.5548673 0.7083568 0.09434398 -0.430138 -0.32107404 0.7412775 0.049827587 0.6043203; -0.7756676 0.34034 -0.16786145 0.50529784 -0.6974864 -0.13994639 -0.19976921 -0.1963243; -0.26765147 -0.66440815 0.40269566 -0.64491284 -0.1650274 0.26684743 0.24802266 -0.12115024; -0.72465456 -0.42832336 0.5313636 -0.4804739 -0.13507439 0.029647877 -0.44190553 0.22368813; 0.48481333 -0.18676175 -0.65955776 0.26108047 0.3435478 0.07323632 -0.5232079 -0.41918445; -0.41689095 0.3558209 0.2108721 0.37487972 -0.29755726 -0.19109961 -0.74913067 0.037819672; -0.6550867 0.5482121 0.2632764 -0.44565493 0.60158175 0.4300911 -0.94353527 0.87244016; -0.60216755 0.2656999 0.06900024 0.42859933 -0.27059677 0.5596646 -0.413605 -0.29823056; -0.81836236 -0.27542457 0.46104535 -0.27372167 0.24206059 0.010957091 -0.88916945 0.7515977; 0.26335722 0.2703245 1.014401 -0.9018361 0.25127536 0.17256503 -1.1224933 0.49207824; -0.3856625 0.2755932 0.7670936 -0.43025774 0.45344985 0.88492185 -1.1936108 0.71900004; 0.45687717 0.3820503 -0.698612 0.62198937 -1.2310563 -0.38793194 -0.14614454 -0.06699204; -0.43654734 0.51904684 -0.25918117 0.71761763 -0.8200454 -0.14142591 0.08458623 -0.080746755; -0.55761063 0.7424243 0.6788876 -0.5164999 0.6582091 0.047975145 -0.4862674 0.6473114], bias = Float32[0.289556; 0.27473128; 0.14113389; 0.32415205; 0.3443449; 0.11965636; 0.011683531; 0.9725912; 0.38026178; 0.15589029; 0.008508337; 0.35841817; 0.4343131; 0.06851576; -0.013669899; 0.31845987; 0.8450926; 1.1298075; 1.1574289; 0.7688242; 0.69021; 1.2370393; 0.644356; 0.90957916; 0.4746754; 0.05912842; 0.30261755; 0.62276167; 0.638722; 0.008331176; -0.11761755; 0.87571514;;]), classifier = (weight = Float32[-1.4314163 0.7583046 1.2342653 1.262362 -0.93869805 0.111096166 -0.26748934 1.2274333], bias = Float32[-0.6359896;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

