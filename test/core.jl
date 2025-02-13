using Functors, Lux, Test

include("test_utils.jl")

rng = get_stable_rng(12345)

@testset "Functors Compatibility" begin
    c = Parallel(+;
        chain=Chain(; dense_1=Dense(2 => 3), dense_2=Dense(3 => 5)),
        dense_3=Dense(5 => 1))

    @test_nowarn fmap(println, c)

    l = Dense(2 => 2)
    new_model = fmap(x -> l, c)
    @test new_model.layers.chain.layers.dense_1 == l
    @test new_model.layers.chain.layers.dense_2 == l
    @test new_model.layers.dense_3 == l
end
