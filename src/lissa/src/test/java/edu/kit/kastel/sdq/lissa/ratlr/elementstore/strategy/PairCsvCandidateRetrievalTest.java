/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.elementstore.strategy;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.dataset.PairCsvDataset;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.utils.Pair;

class PairCsvCandidateRetrievalTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void retrievesOnlyPairsPresentForTheCurrentSource() throws IOException {
        Path path = temporaryDirectory.resolve("pairs.csv");
        Files.writeString(path, """
                NameA,pathA,contextA,propertyA,typeA,NameB,pathB,contextB,propertyB,typeB,Relation
                A1,P1,A1 context,,Part,B1,R,B1 context,,Requirement,0
                A2,P2,A2 context,,Part,B1,R,B1 context,,Requirement,1/Refine
                A2,P2,A2 context,,Part,B2,R,B2 context,,Requirement,0
                """);
        PairCsvDataset dataset = PairCsvDataset.load(path);
        List<Artifact> sourceArtifacts = dataset.artifacts(
                PairCsvDataset.Side.B, Artifact.ArtifactType.REQUIREMENT, PairCsvDataset.ContentConflictStrategy.MERGE);
        List<Artifact> targetArtifacts = dataset.artifacts(
                PairCsvDataset.Side.A,
                Artifact.ArtifactType.SOFTWARE_ARCHITECTURE_MODEL,
                PairCsvDataset.ContentConflictStrategy.MERGE);

        var retrieval = new PairCsvCandidateRetrieval(new ModuleConfiguration(
                "pair_csv_candidates",
                Map.of(
                        "path", path.toString(),
                        "source_side", "B",
                        "target_side", "A",
                        "max_results", "infinity")));
        Element source = element(sourceArtifacts.stream()
                .filter(artifact -> artifact.getIdentifier().endsWith("|B2"))
                .findFirst()
                .orElseThrow());
        List<Pair<Element, float[]>> targets = targetArtifacts.stream()
                .map(artifact -> new Pair<>(element(artifact), new float[] {1.0f, 0.0f}))
                .toList();

        var result = retrieval.findSimilarElements(new Pair<>(source, new float[] {1.0f, 0.0f}), targets);
        assertEquals(1, result.size());
        assertEquals(
                "A2",
                result.getFirst()
                        .first()
                        .getIdentifier()
                        .substring(result.getFirst().first().getIdentifier().lastIndexOf('|') + 1));
    }

    private static Element element(Artifact artifact) {
        return new Element(artifact.getIdentifier(), artifact.getType(), artifact.getContent(), 0, null, true);
    }
}
