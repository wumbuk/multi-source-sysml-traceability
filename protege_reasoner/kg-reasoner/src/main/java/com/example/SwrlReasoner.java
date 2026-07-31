package com.example;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;

import org.swrlapi.core.SWRLRuleEngine;
import org.swrlapi.factory.SWRLAPIFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

// import java.io.ByteArrayInputStream;
import java.io.File;
// import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;

public class SwrlReasoner {
    @SuppressWarnings("null")
    public static void main(String[] args) throws Exception {
        //从stdin读JSON格式数据
        ObjectMapper mapper = new ObjectMapper();
        // String testJson =
        //     "{"
        // + "\"individuals\": ["
        // + "  {\"id\":\"ANode1\",\"class\":\"Activity\",\"data\":{\"Name\":\"Blow Cooled Air Back to Cabin\"}},"
        // + "  {\"id\":\"FRNode1\",\"class\":\"FRNode\",\"data\":{\"Name\":\"Cooling\"}}"
        // + "],"
        // + "\"relations\": ["
        // + "  {\"subject\":\"FRNode1\",\"property\":\"relatesTo\",\"object\":\"ANode1\"}"
        // + "]"
        // + "}";

        JsonNode input = mapper.readTree(System.in);

        // // 用字符串模拟 System.in
        // ByteArrayInputStream testInput =
        //         new ByteArrayInputStream(testJson.getBytes(StandardCharsets.UTF_8));

        // JsonNode input = mapper.readTree(testInput);

        // System.out.println(input.get("individuals").size());


        //准备OWLAPI对象
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        String configuredOntologyPath = System.getenv("TRACE_ONTOLOGY_PATH");
        File file = configuredOntologyPath == null || configuredOntologyPath.trim().isEmpty()
                ? new File("ontology/KG.ttl")
                : new File(configuredOntologyPath);
        //从本地文件加载实体
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(file);
        OWLDataFactory df = manager.getOWLDataFactory();

        String classbaseIRI = "http://www.semanticweb.org/zls/ontologies/2026/0/KG/";

        //根据Type找到class

        String baseIRI = "http://www.semantiweb.org/zls/ontologies/2026/0/kG#";
        //动态创建实例
        ArrayNode individuals = (ArrayNode) input.get("individuals");
        for (JsonNode ind : individuals){
            String id = ind.get("id").asText();
            String clsName = ind .get("class").asText();
            OWLClass cls = df.getOWLClass(IRI.create(classbaseIRI + clsName));

            if (!ontology.containsClassInSignature(cls.getIRI())) {
                throw new RuntimeException("本体中不存在 Class: " + clsName);
            }
            OWLNamedIndividual individual =
                df.getOWLNamedIndividual(IRI.create(baseIRI+id));
            manager.addAxiom(
                ontology,
                df.getOWLClassAssertionAxiom(cls, individual)
            );

            //Data properties
            JsonNode data = ind.get("data");
            if (data!=null){
                data.fields().forEachRemaining(e->{
                    OWLDataProperty prop =
                        df.getOWLDataProperty(IRI.create(baseIRI+e.getKey()));
                    manager.addAxiom(
                        ontology,
                        df.getOWLDataPropertyAssertionAxiom(
                            prop,
                            individual,
                            e.getValue().asText()
                        )
                    );
                });
            }
        }
        //动态创建ObjectProperty（初始关系）
        ArrayNode relations1 = (ArrayNode)input.get("relations");

        for (JsonNode r : relations1) {
            OWLObjectProperty prop =
                df.getOWLObjectProperty(
                    IRI.create(classbaseIRI+r.get("property").asText())
                );

            OWLNamedIndividual subj =
                df.getOWLNamedIndividual(
                    IRI.create(baseIRI+r.get("subject").asText())
                );
            OWLNamedIndividual obj =
                df.getOWLNamedIndividual(
                    IRI.create(baseIRI+r.get("object").asText())
                );
            manager.addAxiom(ontology, df.getOWLObjectPropertyAssertionAxiom(prop, subj, obj));

        }


        /*


        这里插入测试代码









        */





        // 1. 推理前的 ObjectPropertyAssertion 集合
        Set<OWLObjectPropertyAssertionAxiom> before = new HashSet<>();
        ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).forEach(before::add);

        // 2. 运行 SWRL 推理
        SWRLRuleEngine engine = SWRLAPIFactory.createSWRLRuleEngine(ontology);
        engine.infer();

        // 3. 推理后的 ObjectPropertyAssertion 集合（也复制）
        Set<OWLObjectPropertyAssertionAxiom> after = new HashSet<>(
                ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION)
        );

        // System.out.println("====== 推理得到的 Object Properties ======");

        // 推理得到的 Object Properties（after-removeBefore）
        Set<OWLObjectPropertyAssertionAxiom> inferred =new HashSet<>(after);
        //推理后删除推理前的数据
        inferred.removeAll(before);
        ArrayNode out = mapper.createArrayNode();

        for (OWLObjectPropertyAssertionAxiom ax : inferred) {
            ObjectNode o = out.addObject();
            o.put("subject",
                ax.getSubject().asOWLNamedIndividual().getIRI().getShortForm());
            o.put("property",
                ax.getProperty().asOWLObjectProperty().getIRI().getShortForm());
            o.put("object",
                ax.getObject().asOWLNamedIndividual().getIRI().getShortForm());
        }

        ObjectNode result = mapper.createObjectNode();
        result.set("relations", out);

        System.out.println(result.toString());
}

    public static void printIndividualInfo(OWLOntology ontology,
                                       OWLDataFactory df,
                                       OWLNamedIndividual individual) {

        System.out.println("==== Individual: " + individual.getIRI().getShortForm() + " ====");

        // 1. 打印 Class
        ontology.getAxioms(AxiomType.CLASS_ASSERTION)
                .stream()
                .filter(ax -> ax instanceof OWLClassAssertionAxiom)
                .map(ax -> (OWLClassAssertionAxiom) ax)
                .filter(ax -> ax.getIndividual().equals(individual))
                .forEach(ax -> System.out.println("Class: " + ax.getClassExpression().asOWLClass().getIRI().getShortForm()));

        // 2. 打印 DataProperty
        ontology.getAxioms(AxiomType.DATA_PROPERTY_ASSERTION)
                .stream()
                .map(ax -> (OWLDataPropertyAssertionAxiom) ax)
                .filter(ax -> ax.getSubject().equals(individual))
                .forEach(ax -> {
                    String prop = ax.getProperty().asOWLDataProperty().getIRI().getShortForm();
                    String value = ax.getObject().getLiteral();
                    System.out.println("DataProperty: " + prop + " = " + value);
                });

        // 3. 打印 ObjectProperty
        ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION)
                .stream()
                .map(ax -> (OWLObjectPropertyAssertionAxiom) ax)
                .filter(ax -> ax.getSubject().equals(individual))
                .forEach(ax -> {
                    String prop = ax.getProperty().asOWLObjectProperty().getIRI().getShortForm();
                    String target = ax.getObject().asOWLNamedIndividual().getIRI().getShortForm();
                    System.out.println("ObjectProperty: " + prop + " -> " + target);
                });

        System.out.println();
    }


}
