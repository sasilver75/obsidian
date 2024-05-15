Refers to a discrepancy arising between training and inference phases.

For example, at training time for LMs, we have perfect gold tokens forced upon us ([[Teacher Forcing]]), but during normal inference, our models' input includes previously-decoded tokens from the model. Hence, the model doesn't have access to the ground-truth previous tokens and must rely entirely on its own output!

As a result, the model might perform as well during inference because it's never been trained to handle its own errors; as a result, errors can accumulate over the sequence, leading to degraded performance.