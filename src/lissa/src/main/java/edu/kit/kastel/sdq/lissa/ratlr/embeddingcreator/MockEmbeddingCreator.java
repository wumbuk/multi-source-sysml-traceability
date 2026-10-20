/* Licensed under MIT 2025. */
package edu.kit.kastel.sdq.lissa.ratlr.embeddingcreator;

import java.util.List;

import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;

/**
 * A mock implementation of the embedding creator that returns zero vectors for all elements.
 * This class serves two primary purposes:
 * <ol>
 *     <li>Testing: Provides a simple way to test the embedding pipeline without
 *         incurring the computational cost of real embedding generation</li>
 *     <li>Ensemble Mode: Can be used in the Ensemble Mode of LLMs (specified as "classifiers"
 *         in the configuration) for multi-layered classification, where actual embeddings
 *         are not required for the classification process</li>
 * </ol>
 *
 * The implementation assigns a zero vector of length 1 to each element, making it
 * suitable for scenarios where the actual embedding values are not relevant to the
 * classification process.
 */
public class MockEmbeddingCreator extends EmbeddingCreator {
    public MockEmbeddingCreator(ContextStore contextStore) {
        super(contextStore);
    }

    /**
     * Calculates mock embeddings for a list of elements.
     * Each element is assigned a zero vector of length 1.
     *
     * @param elements The list of elements to create mock embeddings for
     * @return A list of zero vectors, one for each input element
     */
    @Override
    public List<float[]> calculateEmbeddings(List<Element> elements) {
        return elements.stream().map(it -> new float[] {0}).toList();
    }
}
