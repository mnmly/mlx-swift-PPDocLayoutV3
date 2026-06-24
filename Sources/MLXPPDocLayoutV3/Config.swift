// Configuration for PP-DocLayoutV3.
//
// PORT FROM: reference/python/configuration_pp_doclayout_v3.py
//            + backbone_config (hgnet_v2, arch "L")

import Foundation

/// HGNetV2 backbone configuration (arch "L" preset, hard-coded to the values
/// shipped in `PaddlePaddle/PP-DocLayoutV3_safetensors`).
public struct HGNetV2Config: Sendable {
    public var stemChannels: [Int] = [3, 32, 48]
    public var stemStrides: [Int] = [2, 1, 1, 2, 1]
    public var embeddingSize: Int = 64
    public var hiddenSizes: [Int] = [256, 512, 1024, 2048]
    public var hiddenAct: String = "relu"
    public var numChannels: Int = 3
    public var useLearnableAffineBlock: Bool = false

    public var stageInChannels: [Int] = [48, 128, 512, 1024]
    public var stageMidChannels: [Int] = [48, 96, 192, 384]
    public var stageOutChannels: [Int] = [128, 512, 1024, 2048]
    public var stageNumBlocks: [Int] = [1, 1, 3, 1]
    public var stageNumLayers: [Int] = [6, 6, 6, 6]
    public var stageDownsample: [Bool] = [false, true, true, true]
    public var stageLightBlock: [Bool] = [false, false, true, true]
    public var stageKernelSize: [Int] = [3, 3, 5, 5]
    public var stageDownsampleStrides: [Int] = [2, 2, 2, 2]

    public init() {}
}

public struct PPDocLayoutV3Config: Decodable, Sendable {
    // --- core dims ---
    public var dModel: Int
    public var encoderHiddenDim: Int
    public var encoderInChannels: [Int]
    public var featStrides: [Int]
    public var encoderFFNDim: Int
    public var encoderAttentionHeads: Int
    public var encoderLayers: Int
    public var encodeProjLayers: [Int]
    public var positionalEncodingTemperature: Int
    public var encoderActivationFunction: String
    public var activationFunction: String
    public var hiddenExpansion: Float
    public var normalizeBefore: Bool

    // --- mask / order extensions ---
    public var maskFeatureChannels: [Int]
    public var x4FeatDim: Int
    public var numPrototypes: Int
    public var maskEnhanced: Bool
    public var globalPointerHeadSize: Int

    // --- decoder ---
    public var decoderInChannels: [Int]
    public var decoderFFNDim: Int
    public var numFeatureLevels: Int
    public var decoderNPoints: Int
    public var decoderLayers: Int
    public var decoderAttentionHeads: Int
    public var decoderActivationFunction: String
    public var numQueries: Int
    public var learnInitialQuery: Bool
    public var anchorImageSize: [Int]?

    // --- misc ---
    public var layerNormEps: Float
    public var batchNormEps: Float
    public var numLabels: Int
    public var id2label: [String: String]

    public var backbone = HGNetV2Config()

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderHiddenDim = "encoder_hidden_dim"
        case encoderInChannels = "encoder_in_channels"
        case featStrides = "feature_strides"
        case encoderFFNDim = "encoder_ffn_dim"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderLayers = "encoder_layers"
        case encodeProjLayers = "encode_proj_layers"
        case positionalEncodingTemperature = "positional_encoding_temperature"
        case encoderActivationFunction = "encoder_activation_function"
        case activationFunction = "activation_function"
        case hiddenExpansion = "hidden_expansion"
        case normalizeBefore = "normalize_before"
        case maskFeatureChannels = "mask_feature_channels"
        case x4FeatDim = "x4_feat_dim"
        case numPrototypes = "num_prototypes"
        case maskEnhanced = "mask_enhanced"
        case globalPointerHeadSize = "global_pointer_head_size"
        case decoderInChannels = "decoder_in_channels"
        case decoderFFNDim = "decoder_ffn_dim"
        case numFeatureLevels = "num_feature_levels"
        case decoderNPoints = "decoder_n_points"
        case decoderLayers = "decoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderActivationFunction = "decoder_activation_function"
        case numQueries = "num_queries"
        case learnInitialQuery = "learn_initial_query"
        case anchorImageSize = "anchor_image_size"
        case layerNormEps = "layer_norm_eps"
        case batchNormEps = "batch_norm_eps"
        case id2label = "id2label"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ def: T) -> T { (try? c.decode(T.self, forKey: k)) ?? def }
        dModel = d(.dModel, 256)
        encoderHiddenDim = d(.encoderHiddenDim, 256)
        encoderInChannels = d(.encoderInChannels, [512, 1024, 2048])
        featStrides = d(.featStrides, [8, 16, 32])
        encoderFFNDim = d(.encoderFFNDim, 1024)
        encoderAttentionHeads = d(.encoderAttentionHeads, 8)
        encoderLayers = d(.encoderLayers, 1)
        encodeProjLayers = d(.encodeProjLayers, [2])
        positionalEncodingTemperature = d(.positionalEncodingTemperature, 10000)
        encoderActivationFunction = d(.encoderActivationFunction, "gelu")
        activationFunction = d(.activationFunction, "silu")
        hiddenExpansion = d(.hiddenExpansion, 1.0)
        normalizeBefore = d(.normalizeBefore, false)
        maskFeatureChannels = d(.maskFeatureChannels, [64, 64])
        x4FeatDim = d(.x4FeatDim, 128)
        numPrototypes = d(.numPrototypes, 32)
        maskEnhanced = d(.maskEnhanced, true)
        globalPointerHeadSize = d(.globalPointerHeadSize, 64)
        decoderInChannels = d(.decoderInChannels, [256, 256, 256])
        decoderFFNDim = d(.decoderFFNDim, 1024)
        numFeatureLevels = d(.numFeatureLevels, 3)
        decoderNPoints = d(.decoderNPoints, 4)
        decoderLayers = d(.decoderLayers, 6)
        decoderAttentionHeads = d(.decoderAttentionHeads, 8)
        decoderActivationFunction = d(.decoderActivationFunction, "relu")
        numQueries = d(.numQueries, 300)
        learnInitialQuery = d(.learnInitialQuery, false)
        anchorImageSize = (try? c.decodeIfPresent([Int].self, forKey: .anchorImageSize)) ?? nil
        layerNormEps = d(.layerNormEps, 1e-5)
        batchNormEps = d(.batchNormEps, 1e-5)
        id2label = d(.id2label, [:])
        numLabels = id2label.isEmpty ? 25 : id2label.count
    }

    public static func load(contentsOf url: URL) throws -> PPDocLayoutV3Config {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PPDocLayoutV3Config.self, from: data)
    }
}
