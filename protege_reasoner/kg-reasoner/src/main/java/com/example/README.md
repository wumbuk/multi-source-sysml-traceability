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

        // //写入DataProperty属性
        // OWLDataProperty nameProp =
        // df.getOWLDataProperty(IRI.create(baseIRI + "Name"));

        // //添加ANode节点名称
        // OWLAxiom ANodenameAxiom =
        //         df.getOWLDataPropertyAssertionAxiom(
        //                 nameProp,
        //                 Actionindividual,
        //                 "Blow Cooled Air Back to Cabin"
        //         );

        // manager.addAxiom(ontology, ANodenameAxiom);
        // //添加FRNode节点名称
        // OWLAxiom FRNodenameAxiom =
        //         df.getOWLDataPropertyAssertionAxiom(
        //             nameProp,
        //             FuncRequireindividual,
        //             "Cooling"
        //         );
        // manager.addAxiom(ontology,FRNodenameAxiom);

        // //添加PATH值
        // OWLDataProperty pathProp = df.getOWLDataProperty(IRI.create(baseIRI + "Path"));
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //         pathProp,
        //         Actionindividual,//活动节点添加
        //         "[1 Problem Domain::2 White Box::3 Functional Analysis::Reach Desired Temperature]"
        // )
        // );
        // manager.addAxiom(ontology,
        //     df.getOWLDataPropertyAssertionAxiom(
        //         pathProp,
        //         FuncRequireindividual, //需求节点添加
        //         "[1 Problem Domain::2 White Box::2 Conceptual Subsystems Communication::VCCU]"
        //     )
        // );

        //添加Descroption值
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

        //添加Type属性值(不是必须)
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

        //添加两个实体之间的relatesTo关系
        // 1) 获取 relatesTo ObjectProperty,这里规则的前缀要和SWRL规则中保持一致
        // OWLObjectProperty relatesToProp =
        //         df.getOWLObjectProperty(IRI.create(classbaseIRI + "relatesTo"));

        // // 2) 添加关系断言：FuncRequireindividual -> Actionindividual
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
        //输出创建的两个实例
        // printIndividualInfo(ontology, df, Actionindividual);
        // printIndividualInfo(ontology, df, FuncRequireindividual);

        // //对于从本地文件解析本体的验证,验证包括classes,个体,Rules
        // System.out.println("===============验证================");
        // System.out.println("Classes: " + ontology.getClassesInSignature().size());
        // System.out.println("Individuals: " + ontology.getIndividualsInSignature().size());
        // System.out.println("SWRL rules: " + ontology.getAxioms(AxiomType.SWRL_RULE).size());

        //23,2,1,类,个体,规则数量均正确,下面进行内容检查
        // System.out.println("====== Classes ======");
        // ontology.getClassesInSignature().forEach(cls -> {
        //     System.out.println(cls.getIRI().getShortForm());
        // });


        以上是测试创建实例功能的代码，插入到SwrlReasoner中使用，


如果需要添加新的规则，
