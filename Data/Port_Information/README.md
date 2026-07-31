关于这里要保存的系统端口信息流，要包含三部分
1.系统块的定义，比如<packagedElement xmi:type="uml:Class" xmi:id="_19_0_2_534018e_1573224071750_518053_42721" name="VCCU">
  系统中端口的定义，<ownedAttribute xmi:type="uml:Port" xmi:id="_19_0_3_534018e_1576680025921_907977_42869" name="p3" visibility="public" aggregation="composite" type="_19_0_3_534018e_1576679926758_30020_42665"/>
  并且要保持缩进，说明端口属于哪个子系统
  格式为：
  <packagedElement xmi:type="uml:Class" xmi:id="_19_0_3_534018e_1583333911778_287119_42741" name="Cooling">
	<ownedAttribute xmi:type="uml:Port" xmi:id="_19_0_3_534018e_1583335395156_334092_42915" name="p3" visibility="public" aggregation="composite" type="_19_0_3_534018e_1576679926758_30020_42665"/>

2.connector的定义
  每个信息流是由哪个端口发出，哪个端口接收，例如：
  <ownedConnector xmi:type="uml:Connector" xmi:id="_19_0_3_534018e_1583335395154_761257_42912" visibility="public">
  <end xmi:type="uml:ConnectorEnd" xmi:id="_19_0_3_534018e_1583335395154_769567_42913" role="_19_0_3_534018e_1576680025921_907977_42869"/>
  <end xmi:type="uml:ConnectorEnd" xmi:id="_19_0_3_534018e_1583335395154_125943_42914" partWithPort="_19_0_3_534018e_1583333905119_835914_42707" role="_19_0_3_534018e_1583335395156_334092_42915"/>
  </ownedConnector>

3.信息流的定义
  <packagedElement xmi:type="uml:InformationFlow" xmi:id="_19_0_3_534018e_1583335403414_162903_42952" name="flow for Air">
  <conveyed xmi:idref="_19_0_2_534018e_1575983285417_72246_42356"/>
  <informationSource xmi:idref="_19_0_2_534018e_1573224071750_518053_42721"/>
  <informationTarget xmi:idref="_19_0_3_534018e_1583333911778_287119_42741"/>
  <realizingConnector xmi:idref="_19_0_3_534018e_1583335395154_761257_42912"/>
  </packagedElement>