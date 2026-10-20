/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.elementstore.strategy;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.dataset.PairCsvDataset;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.utils.Pair;

/** Restricts similarity retrieval to candidate pairs explicitly present in a paired CSV dataset. */
public final class PairCsvCandidateRetrieval implements RetrievalStrategy {
    private final Map<String, Set<String>> candidateTargets;
    private final int maxResults;

    public PairCsvCandidateRetrieval(ModuleConfiguration configuration) {
        PairCsvDataset dataset = PairCsvDataset.load(Path.of(configuration.argumentAsString("path")));
        PairCsvDataset.Side sourceSide = PairCsvDataset.Side.from(configuration.argumentAsString("source_side", "B"));
        PairCsvDataset.Side targetSide = PairCsvDataset.Side.from(configuration.argumentAsString("target_side", "A"));
        this.candidateTargets = dataset.candidateTargets(sourceSide, targetSide);

        String configuredMaximum = configuration.argumentAsString("max_results", "10");
        if (configuredMaximum.equalsIgnoreCase(CosineSimilarity.MAX_RESULTS_INFINITY_ARGUMENT)) {
            this.maxResults = Integer.MAX_VALUE;
        } else {
            this.maxResults = Integer.parseInt(configuredMaximum);
            if (maxResults < 1) {
                throw new IllegalArgumentException("The maximum number of results must be greater than 0.");
            }
        }
    }

    @Override
    public List<Pair<Element, Float>> findSimilarElements(
            Pair<Element, float[]> query, List<Pair<Element, float[]>> allElementsInStore) {
        Set<String> allowed = candidateTargets.get(query.first().getIdentifier());
        if (allowed == null) {
            throw new IllegalArgumentException("Source element has no candidate rows in paired CSV dataset: "
                    + query.first().getIdentifier());
        }

        List<Pair<Element, Float>> similarElements = new ArrayList<>();
        for (Pair<Element, float[]> candidate : allElementsInStore) {
            if (allowed.contains(candidate.first().getIdentifier())) {
                similarElements.add(
                        new Pair<>(candidate.first(), cosineSimilarity(query.second(), candidate.second())));
            }
        }
        if (similarElements.size() != allowed.size()) {
            throw new IllegalArgumentException("Paired CSV candidate set references " + allowed.size()
                    + " targets, but only " + similarElements.size() + " are present in the target store for "
                    + query.first().getIdentifier());
        }
        similarElements.sort((left, right) -> Float.compare(right.second(), left.second()));
        return List.copyOf(similarElements.subList(0, Math.min(maxResults, similarElements.size())));
    }

    private static float cosineSimilarity(float[] queryVector, float[] elementVector) {
        if (queryVector.length != elementVector.length) {
            throw new IllegalArgumentException("The length of the query vector and the element vector must be equal.");
        }
        double dotProduct = 0.0;
        double queryNorm = 0.0;
        double elementNorm = 0.0;
        for (int index = 0; index < queryVector.length; index++) {
            dotProduct += queryVector[index] * elementVector[index];
            queryNorm += queryVector[index] * queryVector[index];
            elementNorm += elementVector[index] * elementVector[index];
        }
        double denominator = Math.sqrt(queryNorm) * Math.sqrt(elementNorm);
        return denominator == 0.0 ? 0.0f : (float) (dotProduct / denominator);
    }
}
