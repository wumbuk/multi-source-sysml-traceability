/* Licensed under MIT 2025. */
package edu.kit.kastel.sdq.lissa.ratlr.postprocessor;

import java.util.Set;

import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.TraceLink;

/**
 * A postprocessor that performs no modifications to trace links.
 * This class serves as a pass-through processor, returning the input trace links
 * without any changes. It is useful in scenarios where no ID transformation is needed
 * or as a default processor when no specific processing is required.
 */
public class IdentityPostprocessor extends TraceLinkIdPostprocessor {

    public IdentityPostprocessor(ContextStore contextStore) {
        super(contextStore);
    }

    /**
     * Returns the input trace links without any modifications.
     * This method simply returns the input set, preserving all trace links exactly as they are.
     *
     * @param traceLinks The set of trace links to process
     * @return The same set of trace links, unmodified
     */
    @Override
    public Set<TraceLink> postprocess(Set<TraceLink> traceLinks) {
        return traceLinks;
    }
}
