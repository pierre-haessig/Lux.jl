


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
((lstm_cell = (weight_i = Float32[-0.8617099 -0.41746527; -0.2545609 -0.75595665; 0.80082566 0.879571; -0.051910702 0.0766955; -0.1489302 -0.61869544; 0.29228586 0.55653924; -0.83112377 0.06285612; 0.080508836 0.059182897; -0.03964443 0.08580193; -0.6592308 0.45296127; 1.1386614 -0.028846165; -0.010067445 0.08946189; -0.10441881 0.24692291; -0.8449612 0.33022285; -0.13243093 -0.2824669; 1.2270906 0.094967514; 0.6751065 -0.6308605; 0.1337152 -0.34881297; 0.47721702 -0.3287928; -0.5622167 0.5162342; 0.5844772 -0.835601; 0.108528465 0.297993; 0.8640756 -0.6402232; 0.94509006 0.61355984; -1.2180481 -0.10157634; 0.48762733 0.6441328; 1.0584478 0.67854786; -0.4578761 -0.17066546; 0.727071 -0.70562476; -0.33273652 0.7480687; -0.2172992 0.61045784; 0.6085898 -0.31582624], weight_h = Float32[-0.510663 -0.052607022 0.27749127 -0.26244292 0.3011389 0.0057322197 -0.68882054 0.5542043; -0.6423588 0.2336679 -0.07004444 0.64413685 0.38700023 0.31292805 0.1316813 -0.05035978; 0.023214774 0.05925412 -0.034522228 0.55422777 -0.7444506 0.31976417 -0.62857485 -0.17019436; 0.03516872 -0.2786051 0.8858645 -0.09060833 0.9110059 0.072432384 0.27516168 0.9070332; -0.3639913 0.47394568 0.7474355 0.300277 0.40201318 0.64011496 -0.3345975 -0.078759424; -0.061680876 -0.33890456 0.33340758 -0.09718059 -0.2919983 -0.22230908 -0.50339186 0.04108233; -0.7787175 0.36799043 0.6960955 0.49063906 -0.80112857 0.6362734 0.036739018 -0.62928426; 0.6796208 0.30480856 1.01812 -1.319461 0.8323077 0.1508899 -0.6975553 1.1549153; -0.14512785 0.47464317 0.39986858 -0.50584924 0.46553305 0.6748142 0.19761162 0.6438947; -0.41791254 -0.18791075 -0.37962398 -0.014123723 0.24647547 0.0031688518 -0.7428138 0.7377413; -0.14375028 0.5506729 -0.07950549 0.5273237 -0.52796614 0.29213116 -0.26364127 -0.3840038; 0.05537054 0.8790385 0.28494555 -0.10549156 0.82662106 0.5980711 0.39123318 0.4978297; 0.65060097 0.4892573 0.37402058 -0.07162972 0.9219713 0.18902576 -0.6481554 0.4428438; -0.10528788 0.44037196 0.08185423 0.077862956 0.60627705 -0.013476268 -0.13111712 -0.43189606; -0.841649 0.3183205 0.09803853 -0.40796527 -0.25439003 0.8134655 -0.35978574 0.41135785; -0.6546861 0.7857179 0.41627082 1.0339079 -0.1003863 0.82163733 -0.11556131 0.75746405; 0.1543253 -0.6347169 -0.05096446 -0.43598723 -0.35186234 -0.20898394 -0.15977356 0.18303582; -0.43591335 0.22607094 0.20064588 0.71222067 0.3961051 -0.35098127 -0.36443612 0.6842029; -0.55122375 0.7091092 0.095555834 -0.4349304 -0.31250176 0.7382125 0.052832328 0.59840983; -0.7739128 0.34114885 -0.16682719 0.50202775 -0.695701 -0.15558977 -0.19824678 -0.19775929; -0.27373934 -0.6682423 0.4069455 -0.643149 -0.17133561 0.26580778 0.24648832 -0.12003723; -0.72751075 -0.42136356 0.52710295 -0.47297388 -0.17268354 0.025547557 -0.44023415 0.21821325; 0.48521858 -0.18443564 -0.6646472 0.29411697 0.3423462 0.06813598 -0.5228851 -0.41360062; -0.40902162 0.3528133 0.20912424 0.36345676 -0.32330793 -0.19936423 -0.74053955 0.010845802; -0.64063275 0.5550311 0.25972834 -0.46348295 0.60499126 0.43691108 -0.925236 0.87113905; -0.6064811 0.27621064 0.0605235 0.43005192 -0.26696193 0.56847477 -0.41500136 -0.3381173; -0.81934786 -0.27449906 0.4641746 -0.27251723 0.22562966 0.014192531 -0.88398105 0.74186355; 0.30035058 0.2669547 1.0115255 -0.904039 0.2444554 0.17604208 -1.0894995 0.46699652; -0.34983486 0.28035244 0.77276504 -0.4405712 0.45167387 0.91810036 -1.1837423 0.69912755; 0.42481816 0.40551567 -0.691262 0.5950554 -1.120953 -0.36473176 -0.17293262 -0.060109843; -0.4282332 0.5152091 -0.240157 0.69405544 -0.8184996 -0.13981098 0.08328746 -0.095393576; -0.54553807 0.76457894 0.7022575 -0.54649603 0.6202318 0.060064342 -0.48658782 0.65834296], bias = Float32[0.29406896; 0.29013303; 0.1333129; 0.32038605; 0.34538686; 0.11529254; 0.010987968; 0.982928; 0.38456228; 0.15334095; 0.00561119; 0.35480288; 0.43182716; 0.06554458; -0.014468816; 0.3082711; 0.84911096; 1.1265159; 1.1538554; 0.77021337; 0.6879448; 1.2413853; 0.65659666; 0.9223942; 0.47872552; 0.046621524; 0.29581386; 0.61346555; 0.63251173; 0.028937135; -0.11961959; 0.8760049;;]), classifier = (weight = Float32[-1.4408381 0.7595659 1.244037 1.2663889 -0.9408289 0.118590035 -0.26396927 1.2255589], bias = Float32[-0.55432385;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

