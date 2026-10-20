/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.preprocessor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;

class SysMLV2PreprocessorTest {
    private static final String MODEL = """
            package VehicleSystems {
                requirement def <'REQ-EB'> EmergencyBrakingRequirement {
                    doc /* Stop the vehicle when an obstacle is detected. */
                    subject vehicle : Vehicle;
                }

                part def Vehicle {
                    doc /* Road vehicle with emergency braking. */
                    part brakeController : BrakeController;
                    port commandPort : CommandPort;
                }

                part def BrakeController {
                    attribute stoppingDistance : ScalarValues::Real;
                }
            }
            """;

    @Test
    void extractsTypedHierarchicalElementsWithStableIds() {
        var configuration = new ModuleConfiguration(
                "model_sysml_v2",
                Map.of(
                        "include_types", "requirement,part",
                        "exclude_types", "package",
                        "include_body", "true",
                        "strict", "true"));
        var preprocessor = new SysMLV2Preprocessor(configuration, new ContextStore());

        List<Element> elements = preprocessor.preprocess(List.of(new Artifact("vehicle.sysml", "sysml model", MODEL)));

        assertEquals(8, elements.size());
        Element requirement = find(elements, "requirement-def:EmergencyBrakingRequirement");
        assertEquals("sysml requirement definition", requirement.getType());
        assertEquals(2, requirement.getGranularity());
        assertTrue(requirement.isCompare());
        assertTrue(requirement.getContent().contains("Short Name: REQ-EB"));
        assertTrue(requirement.getContent().contains("VehicleSystems::EmergencyBrakingRequirement"));

        Element usage = find(elements, "part-def:Vehicle$part:brakeController");
        assertEquals(3, usage.getGranularity());
        assertEquals(
                "Vehicle", usage.getParent().getContent().split("Name: ")[1].split("\\R")[0]);
        assertTrue(usage.isCompare());

        Element port = find(elements, "port:commandPort");
        assertFalse(port.isCompare());

        Element vehicle = find(elements, "part-def:Vehicle");
        assertTrue(vehicle.getContent().contains("Road vehicle with emergency braking"));
        assertFalse(vehicle.getContent().contains("brakeController : BrakeController"));
    }

    @Test
    void strictModeRejectsContentWithoutSupportedDeclarations() {
        var preprocessor =
                new SysMLV2Preprocessor(new ModuleConfiguration("model_sysml_v2", Map.of()), new ContextStore());
        assertThrows(
                IllegalArgumentException.class,
                () -> preprocessor.preprocess(List.of(new Artifact("empty.sysml", "sysml model", "alias X for Y;"))));
    }

    private static Element find(List<Element> elements, String identifierSuffix) {
        return elements.stream()
                .filter(it -> it.getIdentifier().endsWith(identifierSuffix))
                .findFirst()
                .orElseThrow();
    }
}
