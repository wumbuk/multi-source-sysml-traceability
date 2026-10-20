/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ReasoningClassifierTraceDecisionTest {
    @Test
    void parsesPositiveRefineDecisionCaseInsensitively() {
        var decision = ReasoningClassifier.parseTraceDecision(
                "Reasoning first. <trace> YES </trace> <relation_type> refine </relation_type>");

        assertEquals(new ReasoningClassifier.TraceDecision(true, "Refine"), decision);
    }

    @Test
    void parsesPositiveSatisfyDecision() {
        var decision = ReasoningClassifier.parseTraceDecision(
                "<trace>yes</trace>\n<relation_type>Satisfy</relation_type>");

        assertEquals(new ReasoningClassifier.TraceDecision(true, "Satisfy"), decision);
    }

    @Test
    void rejectsInconsistentNegativeType() {
        assertThrows(IllegalArgumentException.class, () -> ReasoningClassifier.parseTraceDecision(
                "<trace>no</trace><relation_type>Refine</relation_type>"));
    }

    @Test
    void rejectsMissingRelationType() {
        assertThrows(IllegalArgumentException.class,
                () -> ReasoningClassifier.parseTraceDecision("<trace>yes</trace>"));
    }

    @Test
    void parsesTagsWhenTheyAppearAfterThinkingText() {
        var decision = ReasoningClassifier.parseTraceDecision(
                "Internal reasoning.\nFinal: <trace>yes</trace> <relation_type>Satisfy</relation_type>");

        assertEquals(new ReasoningClassifier.TraceDecision(true, "Satisfy"), decision);
    }

    @Test
    void ignoresReasoningExamplesFromServerProbe3137() {
        String response = "Example <trace>yes/no</trace><relation_type>...</relation_type>\n"
                + "Alternative <trace>no</trace><relation_type>None</relation_type>\n"
                + "<trace>yes</trace><relation_type>Satisfy</relation_type>\n";
        assertEquals(new ReasoningClassifier.TraceDecision(true, "Satisfy"),
                ReasoningClassifier.parseTraceDecision(response));
    }

    @Test
    void acceptsFinalNegativeAfterPositiveExample() {
        assertEquals(new ReasoningClassifier.TraceDecision(false, "None"),
                ReasoningClassifier.parseTraceDecision("<trace>yes</trace><relation_type>Refine</relation_type>\n"
                        + "<trace>no</trace><relation_type>None</relation_type>"));
    }

    @Test
    void rejectsInvalidFinalAnswersInsteadOfFallingBackToEarlierExample() {
        String example = "<trace>yes</trace><relation_type>Satisfy</relation_type>\n";
        for (String invalid : new String[] {
                "<trace>yes/no</trace><relation_type>Satisfy</relation_type>",
                "<trace>yes</trace><relation_type>None</relation_type>",
                "<trace>no</trace><relation_type>Refine</relation_type>",
                "<trace>yes</trace><relation_type>...</relation_type>",
                "<trace>yes</trace>", "No final answer", ""}) {
            assertThrows(IllegalArgumentException.class,
                    () -> ReasoningClassifier.parseTraceDecision(example + invalid + " trailing"));
            if (!invalid.isEmpty()) {
                assertThrows(IllegalArgumentException.class,
                        () -> ReasoningClassifier.parseTraceDecision(example + invalid));
            }
        }
    }
}
