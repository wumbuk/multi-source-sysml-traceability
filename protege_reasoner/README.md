test.json中存放的是调用jar包进行推理的数据输入格式，
命令为Get-Content .\test.json | java -jar .\kg-reasoner-1.0-SNAPSHOT.jar

如果需要添加新的推理规则，在这个kg-reasoner\build\ttl\路径下添加到KG.ttl
需要在protege本体设计中先去添加新的SWRL推理规则，输出ttl文件，然后替换KG.ttl