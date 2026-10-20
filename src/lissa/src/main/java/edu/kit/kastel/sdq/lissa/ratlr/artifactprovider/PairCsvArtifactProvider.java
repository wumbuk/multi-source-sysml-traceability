/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.artifactprovider;

import java.nio.file.Path;
import java.util.List;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.dataset.PairCsvDataset;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;

/** Provides deduplicated A- or B-side artifacts from a paired trace-relation CSV dataset. */
public final class PairCsvArtifactProvider extends ArtifactProvider {
    private final PairCsvDataset dataset;
    private final PairCsvDataset.Side side;
    private final Artifact.ArtifactType artifactType;
    private final PairCsvDataset.ContentConflictStrategy conflictStrategy;
    private List<Artifact> artifacts;

    public PairCsvArtifactProvider(ModuleConfiguration configuration, ContextStore contextStore) {
        super(contextStore);
        this.dataset = PairCsvDataset.load(Path.of(configuration.argumentAsString("path")));
        this.side = PairCsvDataset.Side.from(configuration.argumentAsString("side"));
        this.artifactType = Artifact.ArtifactType.from(configuration.argumentAsString("artifact_type"));
        this.conflictStrategy = PairCsvDataset.ContentConflictStrategy.from(
                configuration.argumentAsString("content_conflict", "merge"));
    }

    @Override
    public List<Artifact> getArtifacts() {
        if (artifacts == null) {
            artifacts = dataset.artifacts(side, artifactType, conflictStrategy);
        }
        return artifacts;
    }

    @Override
    public Artifact getArtifact(String identifier) {
        return getArtifacts().stream()
                .filter(artifact -> artifact.getIdentifier().equals(identifier))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Artifact not found: " + identifier));
    }
}
