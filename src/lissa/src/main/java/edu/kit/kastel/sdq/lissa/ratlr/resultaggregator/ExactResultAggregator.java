/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.resultaggregator;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import edu.kit.kastel.sdq.lissa.ratlr.classifier.ClassificationResult;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.TraceLink;

/** Keeps trace links at the exact element granularity selected by the classifier. */
public final class ExactResultAggregator extends ResultAggregator {
    public ExactResultAggregator(ContextStore contextStore) {
        super(contextStore);
    }

    @Override
    public Set<TraceLink> aggregate(
            List<Element> sourceElements,
            List<Element> targetElements,
            List<ClassificationResult> classificationResults) {
        Set<TraceLink> traceLinks = new LinkedHashSet<>();
        for (ClassificationResult result : classificationResults) {
            traceLinks.add(new TraceLink(
                    result.source().getIdentifier(), result.target().getIdentifier()));
        }
        return traceLinks;
    }
}
