


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
((lstm_cell = (weight_i = Float32[-0.8619046 -0.4227204; -0.25255463 -0.75635636; 0.80334634 0.8802572; -0.054770302 0.07573275; -0.14660689 -0.62348056; 0.2911879 0.5513008; -0.82909274 0.06579861; 0.079456486 0.05723063; -0.039213896 0.0854006; -0.65977484 0.45131898; 1.151541 -0.027442247; -0.010419951 0.08859963; -0.10582349 0.24704799; -0.8482983 0.3240377; -0.1305866 -0.2786444; 1.2201636 0.08906434; 0.67635834 -0.6304554; 0.1322055 -0.3474945; 0.47339335 -0.32846648; -0.56016773 0.5178137; 0.5814791 -0.84059477; 0.09932742 0.29465586; 0.86528635 -0.63945186; 0.9453325 0.617066; -1.2439287 -0.098093204; 0.4877478 0.6441836; 1.0569606 0.6769346; -0.46631628 -0.17220782; 0.7295198 -0.70702136; -0.33314505 0.74309224; -0.2159369 0.61420757; 0.60538185 -0.31078032], weight_h = Float32[-0.5134295 -0.05288913 0.28349948 -0.25800297 0.30214784 0.0071610715 -0.69875896 0.55379295; -0.64827615 0.23539285 -0.06652127 0.63707757 0.3882717 0.31251192 0.13059849 -0.047928024; 0.02154836 0.059057392 -0.028218454 0.5558947 -0.74720585 0.32092384 -0.6293285 -0.16556506; 0.037486065 -0.27537078 0.8854264 -0.09098601 0.9112283 0.07765408 0.27791095 0.90694636; -0.36720046 0.47387776 0.7730227 0.30843416 0.40310302 0.6406043 -0.34550837 -0.077310786; -0.05748739 -0.33929783 0.33352584 -0.098672956 -0.28620002 -0.22187741 -0.49866268 0.05088275; -0.7817586 0.36882767 0.6970647 0.49072367 -0.80065876 0.6377487 0.033216704 -0.6231806; 0.68222797 0.30421734 1.0274888 -1.3180592 0.82586515 0.15673625 -0.703813 1.1496937; -0.14737771 0.4738432 0.40364787 -0.5041531 0.46427947 0.6761712 0.19514696 0.6413711; -0.4167293 -0.18870132 -0.38041067 -0.013929825 0.2464338 0.00092883565 -0.74991244 0.74992573; -0.14466637 0.54968745 -0.070341654 0.52760863 -0.52924347 0.29165176 -0.26440609 -0.38201213; 0.056167122 0.8796199 0.28476945 -0.10414878 0.82704437 0.6064435 0.39281374 0.49796763; 0.64976925 0.49597886 0.37284696 -0.069527626 0.9207382 0.18741159 -0.6254872 0.44175264; -0.101132445 0.44266096 0.08458276 0.0815439 0.610243 -0.013693569 -0.13460213 -0.40829876; -0.8415269 0.3209074 0.09090024 -0.40749195 -0.25438792 0.79794043 -0.36373344 0.414693; -0.6497467 0.78763384 0.41108364 1.0397427 -0.104017 0.8217102 -0.111851245 0.75531965; 0.15150282 -0.63590676 -0.049686458 -0.43543482 -0.35159132 -0.21501021 -0.16101827 0.1800513; -0.43704608 0.2270487 0.19907406 0.7131235 0.39810714 -0.35049134 -0.3619419 0.68095535; -0.55153376 0.7066657 0.09426595 -0.43435317 -0.31930217 0.73841536 0.052921806 0.59990627; -0.7750271 0.3397604 -0.16828822 0.5030735 -0.69719446 -0.158921 -0.20013697 -0.19805971; -0.2714382 -0.66574 0.40508893 -0.64205027 -0.17113173 0.26214644 0.24914756 -0.11920291; -0.7229749 -0.42667684 0.52927566 -0.47978067 -0.14650233 0.028196853 -0.43917125 0.2222918; 0.4843477 -0.19768085 -0.66204125 0.29172948 0.33980015 0.06855476 -0.5236544 -0.41050214; -0.4131148 0.35268757 0.20695077 0.3686867 -0.296824 -0.19645345 -0.7429168 -0.027080249; -0.64386904 0.5547086 0.26084724 -0.45575824 0.6017989 0.4362844 -0.9341631 0.8701271; -0.60692453 0.2817911 0.064148314 0.42964363 -0.26986712 0.56722295 -0.41524526 -0.32615036; -0.8193797 -0.2738574 0.4617427 -0.2738936 0.22812338 0.01450369 -0.8858172 0.7473616; 0.29606235 0.2679369 1.0149127 -0.89723593 0.23529008 0.17730927 -1.0960742 0.4675844; -0.3514295 0.2820402 0.7731966 -0.43092346 0.44957146 0.918012 -1.1867397 0.70149416; 0.42912966 0.4034311 -0.6908738 0.6008766 -1.1544231 -0.36869135 -0.16573443 -0.05818356; -0.43667716 0.5168223 -0.24244593 0.6961668 -0.82003164 -0.13871166 0.08057787 -0.09186837; -0.5489743 0.758199 0.6951534 -0.5407563 0.624669 0.05853055 -0.48469833 0.65203387], bias = Float32[0.2944335; 0.2914755; 0.13539115; 0.32210085; 0.3457529; 0.116593055; 0.012591104; 0.98036975; 0.3844782; 0.15341687; 0.0050078877; 0.35516706; 0.4319093; 0.06855185; -0.013892492; 0.3076545; 0.84798276; 1.1279905; 1.1542274; 0.7686776; 0.69070643; 1.2391347; 0.6362514; 0.9165304; 0.4779908; 0.04243739; 0.30116722; 0.61114556; 0.63250583; 0.027063807; -0.11982601; 0.8770086;;]), classifier = (weight = Float32[-1.4396608 0.76441693 1.24446 1.260517 -0.93864506 0.11729466 -0.2675603 1.2247397], bias = Float32[-0.580236;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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

