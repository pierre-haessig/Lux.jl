


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
((lstm_cell = (weight_i = Float32[-0.8645544 -0.41841245; -0.25207916 -0.7547759; 0.81168747 0.88783926; -0.066700846 0.07525703; -0.14513296 -0.619903; 0.28359294 0.54727036; -0.83792055 0.079279765; 0.073702954 0.04959968; -0.03945982 0.07613249; -0.66610163 0.45170555; 1.1858019 -0.013657927; -0.022388317 0.0865969; -0.11411733 0.24553534; -0.8571284 0.31405574; -0.12550254 -0.2690197; 1.216186 0.100757025; 0.68417966 -0.6261292; 0.12598325 -0.3402423; 0.45315143 -0.32228717; -0.547681 0.51786333; 0.5830734 -0.8359943; 0.09265227 0.29121536; 0.8746434 -0.6379436; 0.9603383 0.6281959; -1.2206911 -0.10104343; 0.47158208 0.6424618; 1.0546697 0.6924537; -0.44919053 -0.15402606; 0.7470929 -0.697621; -0.34659258 0.71919066; -0.2183535 0.6069183; 0.62822866 -0.29708183], weight_h = Float32[-0.5119644 -0.057690322 0.3374012 -0.25815076 0.29499277 0.010484496 -0.67788047 0.54785734; -0.6730601 0.2081485 -0.054131743 0.62162215 0.3705758 0.30683988 0.13023962 -0.053024895; 0.011463668 0.063821316 -0.024263935 0.56656677 -0.7380901 0.31314257 -0.64217 -0.15109149; 0.040079053 -0.26187766 0.8883501 -0.09537588 0.9118057 0.10142062 0.28511629 0.90798867; -0.36055225 0.4701897 0.78643906 0.32404098 0.40175208 0.6373874 -0.32421473 -0.07784011; -0.055851165 -0.33441222 0.32902482 -0.094043314 -0.28783813 -0.21638033 -0.47956097 0.042540617; -0.79292333 0.36997974 0.68730587 0.53131807 -0.81650734 0.6358595 0.018013828 -0.62798345; 0.69030285 0.30346656 1.0861495 -1.3066758 0.8086647 0.17091618 -0.72750574 1.1415482; -0.1604504 0.47147444 0.42687243 -0.5008807 0.4604272 0.68055326 0.17249116 0.63834625; -0.42004082 -0.18495178 -0.38506886 -0.0044060964 0.23490274 -0.004770874 -0.7681005 0.6962108; -0.1525128 0.5503276 -0.01559711 0.53011435 -0.5190394 0.29650536 -0.26998448 -0.37606966; 0.060001608 0.8846909 0.29046053 -0.10944171 0.83021295 0.6500385 0.39272878 0.5005288; 0.65195197 0.48237127 0.3756866 -0.07069138 0.9220844 0.16415153 -0.66454893 0.44334215; -0.09406569 0.44510838 0.08308135 0.10168988 0.60732096 -0.015389085 -0.14999834 -0.4144995; -0.8317472 0.32819012 0.08293498 -0.3936638 -0.2652863 0.8029368 -0.37682986 0.4119057; -0.63428247 0.7861262 0.41578126 1.058289 -0.0799148 0.8133453 -0.09896082 0.7640453; 0.14635369 -0.6442048 -0.055840146 -0.44463643 -0.35269228 -0.22495748 -0.16709016 0.1725227; -0.44445518 0.23324373 0.19264604 0.7207451 0.39363545 -0.35871828 -0.36225557 0.64680666; -0.5572512 0.7055623 0.09194274 -0.4264535 -0.32312286 0.7439799 0.047579236 0.6043455; -0.7780797 0.33960354 -0.16884008 0.50661594 -0.6979809 -0.111654416 -0.20061587 -0.19574715; -0.26673034 -0.6622796 0.3917475 -0.6410631 -0.17020509 0.26480317 0.25317958 -0.11218122; -0.7192674 -0.43559653 0.53213274 -0.48755178 -0.12855662 0.034444883 -0.4445781 0.22276856; 0.48544165 -0.19085425 -0.66055894 0.2462905 0.34078777 0.0739733 -0.5226707 -0.41123226; -0.42272332 0.35800147 0.22619799 0.3825564 -0.29901764 -0.1858298 -0.7520514 0.0805302; -0.64916664 0.5476138 0.27093524 -0.44963038 0.6000063 0.4320277 -0.95853263 0.8710149; -0.60267895 0.25524998 0.075137034 0.42976886 -0.26640645 0.55763495 -0.4134202 -0.3050388; -0.81805384 -0.2744045 0.46161243 -0.28429243 0.2655055 0.010439282 -0.8960662 0.75457865; 0.25688532 0.27056074 1.0167953 -0.91250616 0.25787994 0.18385707 -1.1386364 0.5101502; -0.37380934 0.27353433 0.7698418 -0.4444359 0.45055354 0.89036256 -1.196947 0.7122958; 0.45729592 0.3798327 -0.69664794 0.64924604 -1.2246342 -0.38672885 -0.15438536 -0.07147441; -0.42745635 0.5135636 -0.25303546 0.7441279 -0.8160093 -0.13001466 0.0908314 -0.08371936; -0.55031615 0.74379057 0.6806834 -0.5280279 0.6466935 0.054152347 -0.5004897 0.6419152], bias = Float32[0.28809902; 0.2644113; 0.14757867; 0.32613942; 0.34441087; 0.12334754; 0.008780591; 0.97329044; 0.38029575; 0.1583503; 0.009700807; 0.35832343; 0.43113363; 0.06998329; -0.015288656; 0.3230465; 0.84076107; 1.1338831; 1.1603292; 0.7681968; 0.6897829; 1.2308596; 0.634439; 0.9086202; 0.47603935; 0.052309956; 0.3069106; 0.62572694; 0.6348132; 0.0059889485; -0.11799208; 0.88171005;;]), classifier = (weight = Float32[-1.4338653 0.76035446 1.2308856 1.2656682 -0.9368105 0.11145292 -0.26131856 1.2195423], bias = Float32[-0.6118901;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

