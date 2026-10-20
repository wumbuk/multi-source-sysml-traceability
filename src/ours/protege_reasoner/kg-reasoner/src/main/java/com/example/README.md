# OWL Individual Construction Example

English | [中文](README_zh-CN.md)

The following commented Java snippet demonstrates creating individuals and their properties in `SwrlReasoner`. Insert the relevant statements into that class when testing individual construction; this is a source-level example rather than a standalone program.

```java
// String idValue1 = "ANode1";
        // String idValue2 = "FRNode1";


        // OWLNamedIndividual Actionindividual=
        //                     df.getOWLNamedIndividual(IRI.create(baseIRI+idValue1));
        // OWLNamedIndividual  FuncRequireindividual =
        //                     df.getOWLNamedIndividual(IRI.create(baseIRI+idValue2));
        // manager.addAxiom(ontology,
        // df.getOWLClassAssertionAxiom(ActionClass, Actionindividual));

        // manager.addAxiom(ontology,
        // df.getOWLClassAssertionAxiom(FuncRequireClass, FuncRequireindividual));

        // //Set data properties
        // OWLDataProperty nameProp =
        // df.getOWLDataProperty(IRI.create(baseIRI + "Name"));

        // //Set the ANode name
        // OWLAxiom ANodenameAxiom =
        //         df.getOWLDataPropertyAssertionAxiom(
        //                 nameProp,
        //                 Actionindividual,
        //                 "Blow Cooled Air Back to Cabin"
        //         );

        // manager.addAxiom(ontology, ANodenameAxiom);
        // //Set the FRNode name
        // OWLAxiom FRNodenameAxiom =
        //         df.getOWLDataPropertyAssertionAxiom(
        //             nameProp,
        //             FuncRequireindividual,
        //             "Cooling"
        //         );
        // manager.addAxiom(ontology,FRNodenameAxiom);

        // //Set Path values
        // OWLDataProperty pathProp = df.getOWLDataProperty(IRI.create(baseIRI + "Path"));
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //         pathProp,
        //         Actionindividual,//Set the activity node path
        //         "[1 Problem Domain::2 White Box::3 Functional Analysis::Reach Desired Temperature]"
        // )
        // );
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //         pathProp,
        //         FuncRequireindividual, //Set the requirement node path
        //         "[1 Problem Domain::2 White Box::2 Conceptual Subsystems Communication::VCCU]"
        //     )
        // );

        //Set Description values
        // OWLDataProperty descProp =
        // df.getOWLDataProperty(IRI.create(baseIRI + "Description"));

        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //             descProp,
        //             Actionindividual,
        //             "Blows cooled air from the white box back into the cabin to achieve the desired temperature..."
        //     )
        // );
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //             descProp,
        //             FuncRequireindividual,
        //             "The Cooling block represents a subsystem responsible for thermal regulation within the VCCU, ensuring optimal operating temperatures through controlled heat dissipation."

        //     )
        // );

        //Set optional Type values
        // OWLDataProperty typeProp =
        // df.getOWLDataProperty(IRI.create(baseIRI + "Type"));
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //             typeProp,
        //             Actionindividual,
        //             "Activity"
        //     )
        // );
        //         df.getOWLDataProperty(IRI.create(baseIRI + "Type"));
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //             typeProp,
        //             FuncRequireindividual,
        //             "Functional_Requirement"
        //     )
        // );

        //Add a relatesTo relation between the two individuals
        // 1) Get the relatesTo object property; its IRI prefix must match the SWRL rules
        // OWLObjectProperty relatesToProp =
        //         df.getOWLObjectProperty(IRI.create(classbaseIRI + "relatesTo"));

        // // 2) Add the assertion: FuncRequireindividual -> Actionindividual
        // OWLAxiom relatesAxiom =
        //         df.getOWLObjectPropertyAssertionAxiom(
        //                 relatesToProp,
        //                 FuncRequireindividual,
        //                 Actionindividual
        //         );

        // manager.addAxiom(ontology, relatesAxiom);
        // System.out.println("=====SWRL Rules======");
        // ontology.getAxioms(AxiomType.SWRL_RULE).forEach(rule -> {
        //     System.out.println(rule);
        // });
        //Print the two created individuals
        // printIndividualInfo(ontology, df, Actionindividual);
        // printIndividualInfo(ontology, df, FuncRequireindividual);

        // //Inspect the ontology loaded from the local file: classes, individuals, and rules
        // System.out.println("===============Validation================");
        // System.out.println("Classes: " + ontology.getClassesInSignature().size());
        // System.out.println("Individuals: " + ontology.getIndividualsInSignature().size());
        // System.out.println("SWRL rules: " + ontology.getAxioms(AxiomType.SWRL_RULE).size());

        //For the original test fixture, the class, individual, and rule counts were 23, 2, and 1; inspect their contents next
        // System.out.println("====== Classes ======");
        // ontology.getClassesInSignature().forEach(cls -> {
        //     System.out.println(cls.getIRI().getShortForm());
        // });
```

For SWRL rule editing and build commands, see the [reasoner guide](../../../../../../README.md).
