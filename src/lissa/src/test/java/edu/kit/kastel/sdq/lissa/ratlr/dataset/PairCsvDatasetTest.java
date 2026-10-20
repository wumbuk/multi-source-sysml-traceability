/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.dataset;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;

class PairCsvDatasetTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void loadsBothSidesCandidatesAndPositiveRelations() throws IOException {
        Path datasetPath = writeDataset();
        PairCsvDataset dataset = PairCsvDataset.load(datasetPath);

        var targets = dataset.artifacts(
                PairCsvDataset.Side.A,
                Artifact.ArtifactType.SOFTWARE_ARCHITECTURE_MODEL,
                PairCsvDataset.ContentConflictStrategy.MERGE);
        var sources = dataset.artifacts(
                PairCsvDataset.Side.B, Artifact.ArtifactType.REQUIREMENT, PairCsvDataset.ContentConflictStrategy.MERGE);

        assertEquals(3, dataset.size());
        assertEquals(2, targets.size());
        assertEquals(2, sources.size());
        assertTrue(
                targets.stream().anyMatch(artifact -> artifact.getIdentifier().contains("Element%2C One")));
        Artifact merged = targets.stream()
                .filter(artifact -> artifact.getIdentifier().contains("Element%2C One"))
                .findFirst()
                .orElseThrow();
        assertTrue(merged.getContent().contains("Context 1: First, context"));
        assertTrue(merged.getContent().contains("Context 2: Alternative context"));

        var candidates = dataset.candidateTargets(PairCsvDataset.Side.B, PairCsvDataset.Side.A);
        assertEquals(
                2,
                candidates.values().stream().mapToInt(java.util.Set::size).max().orElseThrow());
        assertEquals(
                2,
                dataset.positiveTraceLinks(PairCsvDataset.Side.B, PairCsvDataset.Side.A, "Relation", "1")
                        .size());
    }

    @Test
    void errorStrategyRejectsConflictingContent() throws IOException {
        PairCsvDataset dataset = PairCsvDataset.load(writeDataset());
        assertThrows(
                IllegalArgumentException.class,
                () -> dataset.artifacts(
                        PairCsvDataset.Side.A,
                        Artifact.ArtifactType.SOFTWARE_ARCHITECTURE_MODEL,
                        PairCsvDataset.ContentConflictStrategy.ERROR));
    }

    private Path writeDataset() throws IOException {
        Path path = temporaryDirectory.resolve("pairs.csv");
        Files.writeString(path, """
                NameA,pathA,contextA,propertyA,typeA,NameB,pathB,contextB,propertyB,typeB,Relation
                "Element, One",Model::One,"First, context",p1,Part,Req1,Requirements,Requirement one,,Requirement,1/Refine
                ElementTwo,Model::Two,Second context,p2,Port,Req1,Requirements,Requirement one,,Requirement,0
                "Element, One",Model::One,Alternative context,p1,Part,Req2,Requirements,Requirement two,,Requirement,1/Satisfy
                """);
        return path;
    }
}
