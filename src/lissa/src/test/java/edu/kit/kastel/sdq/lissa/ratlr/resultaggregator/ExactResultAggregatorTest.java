/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.resultaggregator;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import edu.kit.kastel.sdq.lissa.ratlr.classifier.ClassificationResult;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.TraceLink;

class ExactResultAggregatorTest {
    @Test
    void retainsExactElementIdentifiersAcrossDifferentDepths() {
        Element source = new Element("REQ-1", "requirement", "Requirement", 0, null, true);
        Element model = new Element("model.sysml", "sysml model", "Model", 0, null, false);
        Element packageElement = new Element("model.sysml$package:System", "sysml package", "System", 1, model, false);
        Element target = new Element(
                "model.sysml$package:System$part-def:Controller",
                "sysml part definition",
                "Controller",
                2,
                packageElement,
                true);

        var aggregator = new ExactResultAggregator(new ContextStore());
        var links = aggregator.aggregate(
                List.of(source),
                List.of(model, packageElement, target),
                List.of(ClassificationResult.of(source, target)));

        assertEquals(Set.of(new TraceLink("REQ-1", target.getIdentifier())), links);
    }
}
