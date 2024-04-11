# Deep-Head
Our implementation of this new novel Transformer would involve increasing the number of attention heads in earlier model layers, for instance 32 heads. We would then decrease the number of heads as the model goes deeper in layers. This mechanism would allow the model to learn a hierarchy of feature representations by first learning smaller segments of samples which would learn more localized features. Then as the number of heads decrease, the model would be able to learn more global features in a bottom-up approach. The different representations being learned at different levels will be aggregated from Transformer to Transformer via a skip connection from the final cross attention score matrix of one transformer to the final cross attentions score of the next transformer and will be added to those values and normalized across the batch. Because the final tensors for all Transformers will have the same dimensionality, it allows us to aggregate the feature hierarchies learned at different levels through skip connections, and create more robust feature representations.
