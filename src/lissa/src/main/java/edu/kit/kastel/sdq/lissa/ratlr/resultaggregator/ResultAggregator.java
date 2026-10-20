/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.resultaggregator;

import java.util.List;
import java.util.Objects;
import java.util.Set;

import edu.kit.kastel.sdq.lissa.ratlr.classifier.ClassificationResult;
import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.TraceLink;

/**
 * Abstract base class for result aggregators that combine classification results into trace links.
 * Result aggregators are responsible for:
 * <ul>
 *     <li>Taking classification results from multiple elements</li>
 *     <li>Combining these results based on specific rules</li>
 *     <li>Creating trace links between source and target elements</li>
 * </ul>
 * <p>
 * All result aggregators have access to a shared {@link edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore} via the protected {@code contextStore} field,
 * which is initialized in the constructor and available to all subclasses.
 * Subclasses should not duplicate context handling or Javadoc for the context parameter.
 * </p>
 * The class supports various types of aggregators:
 * <ul>
 *     <li>any_connection: Creates trace links when any element in a group has a positive classification</li>
 *     <li>exact: Keeps the exact classified element identifiers without changing granularity</li>
 * </ul>
 *
 * Each aggregator type implements its own strategy for combining classification results
 * and creating trace links between source and target elements.
 */
public abstract class ResultAggregator {
    /**
     * The shared context store for pipeline components.
     * Available to all subclasses for accessing shared context.
     */
    protected final ContextStore contextStore;

    /**
     * Creates a new result aggregator with the specified context store.
     *
     * @param contextStore The shared context store for pipeline components
     */
    protected ResultAggregator(ContextStore contextStore) {
        this.contextStore = Objects.requireNonNull(contextStore);
    }

    /**
     * Aggregates classification results into a set of trace links.
     * This method:
     * <ol>
     *     <li>Takes lists of source and target elements</li>
     *     <li>Processes the classification results for these elements</li>
     *     <li>Applies the aggregator's specific rules to combine results</li>
     *     <li>Creates trace links between elements based on the combined results</li>
     * </ol>
     *
     * @param sourceElements The list of source elements
     * @param targetElements The list of target elements
     * @param classificationResults The classification results to aggregate
     * @return A set of trace links created from the aggregated results
     */
    public abstract Set<TraceLink> aggregate(
            List<Element> sourceElements,
            List<Element> targetElements,
            List<ClassificationResult> classificationResults);

    /**
     * Creates a result aggregator instance based on the module configuration.
     * The type of aggregator is determined by the configuration name.
     *
     * @param configuration The module configuration specifying the type of aggregator
     * @param contextStore The shared context store for pipeline components
     * @return A new result aggregator instance
     * @throws IllegalStateException if the configuration name is not recognized
     */
    public static ResultAggregator createResultAggregator(
            ModuleConfiguration configuration, ContextStore contextStore) {
        return switch (configuration.name()) {
            case "any_connection" -> new AnyResultAggregator(configuration, contextStore);
            case "exact" -> new ExactResultAggregator(contextStore);
            default -> throw new IllegalStateException("Unexpected value: " + configuration.name());
        };
    }
}
