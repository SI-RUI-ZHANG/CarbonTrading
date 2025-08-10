# **中国经济指标发布滞后性分析及其在日度投资模型中的应用**

## **I. 执行摘要：发布滞后性概述**

本报告旨在精确确定中国各项经济与金融数据（月度及季度）的典型发布滞后时间，以便于将这些历史数据统一调整为日度频率，供投资模型开发使用。在回溯测试投资模型时，使用数据发布前的“未来”信息会导致前视偏差，进而产生错误的模型评估结果。因此，准确识别并应用这些滞后性对于确保模型有效性至关重要。

本分析通过查阅官方发布日历、统计机构（国家级和省级）的新闻稿以及标明数据获取时间的财经新闻报道，来确定各项数据的发布滞后性。核心关注点在于数据首次向公众正式披露的时间点。

**表1：主要数据类别建议发布滞后性总结**

| 数据类别 | 主要发布机构 | 常规发布模式（例如：M+1月中、M+1月末） | 建议滞后范围（周期结束后台历日） |
| :---- | :---- | :---- | :---- |
| 国家层面月度工业指标 | 国家统计局 (NBS) | M+1 月中旬（约15-18日） | 18天 |
| 国家层面月度宏观指标 | 国家统计局 (NBS)、中国人民银行 (PBOC) | M+1 月上旬至中旬（PMI为M月末/M+1月初） | PMI: 2天; CPI: 11天; 社融: 16天 |
| 省级层面月度指标 | 省级统计局、省发改委/能源局 | M+1 月中下旬至M+2月初 | 25天 (工业增加值); 12-45天 (用电量，视省份而定) |
| 国家层面季度GDP | 国家统计局 (NBS) | Q+1 月中下旬（约16-20日） | 20天 |
| 省级层面季度GDP | 省级统计局 | Q+1 月下旬 | 23-25天 |

## **II. 国家层面月度经济指标：发布滞后性分析**

本章节将详细阐述所有M类国家层面数据的发布滞后性。针对每项指标，我们将确定其发布机构，基于所提供的材料分析其发布日程，并给出具体的滞后建议。

### **A. 工业部门指标（主要由国家统计局 \- NBS发布）**

国家统计局是中国官方工业统计数据的主要来源，其发布的日程表是确定滞后性的关键文件 1。工业数据通常在月中发布。

**1\. 01\_中国\_工业增加值\_可比价\_规模以上工业企业\_当月同比**

* **发布机构：** 国家统计局 (NBS)。  
* **发布模式：** 通常作为“国民经济运行情况”新闻发布会和数据发布的一部分公布，该发布同时包含零售销售、固定资产投资等其他指标。国家统计局的发布日历显示，此数据归类于“规模以上工业生产月度报告”和“国民经济运行情况”下 1。例如，1月、3月至9月、11月、12月的数据通常在次月的14日至19日之间发布。2月份数据通常与1月份数据合并，在3月中旬发布。10月份数据有时会稍晚，在20日左右发布。  
* **支持材料：** 1 (第1行和第7行), 2。3明确指出“工业增加值（同比，左轴）”约在15日发布。2显示“2025年3月份工业生产运行情况”于2025年4月17日发布。  
* **滞后计算：** 若M月数据在M+1月的15-17日左右发布，则滞后时间约为M月底后的15-17天。  
* **建议滞后：** **参考月末后18个日历日**。这为数据完全传播和捕获提供了一个缓冲。对于3月份发布的1-2月合并数据，1月份数据的滞后约为47天（1月底至3月中），2月份数据滞后约为18天（2月底至3月中）。使用者应从3月份的发布日期开始，将该值应用于这两个月份。  
  值得注意的是，工业增加值与一系列其他关键月度指标（如社会消费品零售总额、固定资产投资）一同发布 1。国家统计局的这种协调发布意味着市场会同时接收到关于经济状况的全面快照。这并非随机的数据集合，而是为提供整体视角而精心策划的。这种同步发布可能会比各项数据零星发布更能广泛地影响市场预期和反应。因此，尽管我们计算的是单个指标的滞后，但这些指标的“信息事件”是聚集的。对于模型而言，这表明由于这些指标的同步发布，市场对它们的信号处理在某种程度上是相关的。所选的18天滞后应统一应用于国家统计局发布的这一“月中数据包”。

**2\. 主要工业产品产量 (NBS): 01\_中国\_产量\_原油加工量\_当月值, 01\_中国\_产量\_原煤\_当月值, 01\_中国\_产量\_水泥\_当月值, 01\_中国\_产量\_粗钢\_当月值**

* **发布机构：** 国家统计局 (NBS)。  
* **发布模式：** 这些具体的产量数据通常是“能源生产情况月度报告” 1 或更广泛的“规模以上工业生产月度报告” 1 的一部分。发布日期与工业增加值一致。2也显示“2025年3月份能源生产情况”与“工业生产运行情况”同在2025年4月17日发布。  
* **支持材料：** 1 (第7行和第8行), 2。材料46显示这些数据点被追踪，但不总是指明NBS的确切发布日期，这强化了对NBS日历的依赖。  
* **滞后计算：** 与工业增加值相同。  
* **建议滞后：** **参考月末后18个日历日**。（1-2月合并数据的处理方式同上）。  
  虽然工业增加值（IVA）的同比增长是关注的焦点，但这些关键商品（煤炭、钢铁、水泥、原油加工）的绝对产量为核心工业和建筑部门的实体经济活动提供了更直接的衡量标准。这些数据与IVA同时发布，使得分析师能够更深入地探究工业表现的驱动因素。IVA是一个增加值的概念，一定程度上反映了增长，但也包含了价格变动（尽管使用了“可比价”）。而“产量”数据是物理量，较少受到价格扭曲的影响，直接反映生产水平。同时获得这两类数据 1，使分析师能够辨别IVA增长是由广泛扩张驱动，还是集中在特定的重工业领域，或者是否存在分歧（例如，IVA上升但钢铁产量下降，表明工业结构发生转变）。模型可以将这些实物产量数据用作特定行业乃至更广泛经济健康状况的领先指标，可能提供与汇总IVA不同的信号。18天的滞后适用于两者，使得这种组合分析从同一时间点成为可能。

### **B. 能源部门指标**

能源数据来自国家统计局（生产）和国家能源局（NEA）（消费）。关键在于确定*最早的公开数据*。

**1\. 01\_中国\_发电量\_火电\_当月值**

* **发布机构：** 国家统计局 (NBS)。  
* **发布模式：** 包含在国家统计局的“能源生产情况月度报告”中 1。  
* **支持材料：** 1 (第8行), 2。  
* **滞后计算：** 与国家统计局其他月中发布数据一致。  
* **建议滞后：** **参考月末后18个日历日**。

**2\. 01\_中国\_全社会用电量\_当月值**

* **发布机构：** 国家能源局 (NEA) 通常是*最早的综合性*月度数据的首要来源，尽管国家统计局也报告电力数据。  
* **发布模式：** 国家能源局的发布通常在参考月份后一个月的月中至第三周。  
  * 2024年1月数据：2025年1月20日发布 4 \- 这些材料似乎显示的是2024年*年度*数据在2025年1月发布，这对于年度总结是典型的。需查找月度规律。  
  * 2024年2月数据：3月18日发布 7。滞后约19天（2024年2月有29天）。  
  * 2024年3月数据：4月17日发布 9。滞后约17天。  
  * 2024年4月数据：5月17日发布 11。滞后约17天。  
  * 2025年3月数据：4月18日发布 12。滞后约18天。  
  * 50 (来自2012年) 指出“每月14日本局在门户网站上发布月度全社会用电量等电力行业重要数据”，表明历史上存在月中发布的模式。  
* **支持材料：** 4。  
* **NBS 与 NEA对比：** 1 (NBS日历) 第8行“能源生产情况月度报告”于月中发布（例如1月数据于1月17日）。52和1询问此NBS报告是否包含“全社会用电量”。虽然NBS涵盖*发电量*，但NEA通常被认为是*全社会用电量*的来源，发布时间略有不同。用户文件名指定“全社会用电量”，这与NEA的典型报告更为一致。  
* **滞后计算：** 根据7\-11，发布通常在M+1月的17日至20日之间。  
* **建议滞后：** **参考月末后20个日历日**。这考虑了与NBS部分主要数据相比稍晚的发布，并确保NEA的综合消费数据已公布。  
  尽管NBS和NEA都涉及电力数据，但NEA是“全社会用电量”的权威来源，这是一个需求侧指标。NBS的“能源生产情况月度报告”更侧重于供应侧（发电量）。发布时间的微小差异（NEA可能比NBS能源生产数据晚几天）是合乎逻辑的。用户明确要求“全社会用电量”。NBS日历 1 列出了“能源生产情况月度报告”和“发电量\_火电”，这些是供应侧数据。NEA的新闻稿 4 一贯提及“全社会用电量”。NEA消费数据的发布日期（例如3月数据在4月17/18日，4月数据在5月17日）通常比NBS能源生产报告（例如3月能源生产在4月17日 2）晚一两天或一致。因此，对于将电力消费作为需求指标进行建模，采用约20天滞后的NEA数据是合适的。NBS的发电量数据（约18天滞后）可以在供应侧提供更早的背景信息。

### **C. 宏观经济指标 (NBS)**

**1\. 01\_中国\_CPI\_当月同比**

* **发布机构：** 国家统计局 (NBS)。  
* **发布模式：** CPI是最早发布的关键月度指标之一。NBS日历 1 显示，通常在次月的9日至10日发布上月数据。10月份数据例外，约在15日发布。  
* **支持材料：** 1 (第5行), 53。  
* **滞后计算：** 若M月数据在M+1月的9日/10日发布，则滞后9-10天。  
* **建议滞后：** **参考月末后11个日历日**。这提供了一个小的缓冲。

**2\. 01\_中国\_制造业PMI**

* **发布机构：** 国家统计局 (NBS) 和中国物流与采购联合会 (CFLP)。财新也发布PMI，但用户文件名“中国\_制造业PMI”可能指的是官方NBS版本。  
* **发布模式：** PMI通常在当月最后一天或下月第一天发布。NBS日历 1 显示：  
  * 1月数据：1月27日（若为周末则调整，如“31/一”表示31日，周一）  
  * 3月数据：3月31日（若31日为周末则为4月1日）  
  * 4月数据：4月30日  
  * 5月数据：5月31日  
  * 通常是M月最后一天或M+1月第一个工作日。2显示“2025年4月份采购经理指数”于2025年5月1日发布。  
* **支持材料：** 1 (第4行), 2。17。  
* **滞后计算：** 若在M月最后一天或M+1月1日发布，则参考期结束后的滞后实际上是0-1天。  
* **建议滞后：** **参考月末后2个日历日**。这确保了即使在M+1月第一个工作日发布也能被捕获，并考虑了处理时间。这是所有月度指标中最早的。  
  官方NBS PMI发布非常及时 1，使其成为当月经济健康状况的首批信号之一。这种及时性使其成为市场参与者的真正“领先”指标。用户应注意，还有一个广受关注的财新制造业PMI 17，它更侧重于中小型企业，通常在官方PMI发布后一两天发布。选择使用哪个PMI（或两者都用）取决于模型的具体目标。用户的文件名暗示是官方版本。官方NBS PMI涵盖较大型、通常是国有的企业，而财新PMI则更多地涵盖中小企业和出口导向型企业。它们有时会出现分歧，提供不同的经济信号。官方PMI建议的2天滞后是合适的。如果用户还想纳入财新PMI，其滞后会稍长（例如，月末后3-4天）。两个PMI的存在为更细致的分析（例如，将分歧视为一种信号）提供了机会。

### **D. 金融指标**

**1\. 01\_中国\_社会融资规模\_当月值**

* **发布机构：** 中国人民银行 (PBOC)。  
* **发布模式：** 中国人民银行的金融统计数据，包括社会融资总量 (TSF)、M2和新增贷款，通常在参考月份后一个月的9日至15日之间发布。没有像国家统计局那样的固定日历，但有一个一致的窗口期。18（中国人民银行2025年4月份金融统计数据报告）并未给出*该报告*的发布日期，但显示了*截至4月底*的数据。新闻报道通常是获取中国人民银行数据发布时间的最佳指南。56确认中国人民银行为数据来源。  
* **支持材料：** 18。44。  
* **滞后计算：** 若在M+1月的9-15日之间发布。  
* **建议滞后：** **参考月末后16个日历日**。这是典型发布窗口内的一个保守估计，确保数据被捕获。  
  中国人民银行的月度金融统计数据发布（包括社会融资规模、M1、M2、贷款）不像国家统计局那样遵循预先公布的固定日期，而是有一个典型的窗口期（通常是M+1月的9日至15日）18。这个数据包对于理解信贷增长和货币状况至关重要。与国家统计局 1 不同，材料中没有精确的中国人民银行日历。这种“窗口期”方法对于央行金融数据而言很常见。由于数据高度市场敏感，因此窗口期内的确切发布日期可能会引起市场预期。模型应使用此典型窗口期末尾（例如16天）的一致滞后，以确保数据已可用。中国人民银行发布的丰富数据（不仅是社会融资规模）为金融模型提供了许多变量。

**表2：国家层面月度指标 \- 发布滞后性**

| 指标名称 | 发布机构 | 典型发布模式 (M+1月日或规则) | 建议滞后 (月末后台历日) | 支持材料 |
| :---- | :---- | :---- | :---- | :---- |
| 工业增加值 (规模以上) | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 原油加工量 | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 原煤产量 | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 水泥产量 | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 粗钢产量 | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 火电发电量 | NBS | 14-19日 (10月约20日) | 18 | 1 |
| 全社会用电量 | NEA | 17-20日 | 20 | 7\-11 |
| CPI同比 | NBS | 9-10日 (10月约15日) | 11 | 1 |
| 制造业PMI | NBS | M月末或M+1月1日 | 2 | 1 |
| 社会融资规模 | PBOC | 9-15日 | 16 | 18 |

## **III. 省级层面月度经济指标：发布滞后性分析**

本部分将分析省级数据的发布滞后性，这通常需要更多地依赖新闻报道和对省级统计实践的普遍认知。省级数据通常滞后于国家数据。

### **A. 广东省**

省级统计局发布其自身数据。这有时是独立发布，有时作为更广泛的“经济运行情况”报告的一部分。通常没有像国家统计局那样的固定公开日历。

**1\. 01\_广东\_工业增加值\_可比价\_规模以上工业企业\_当月同比**

* **发布机构：** 广东省统计局。  
* **发布模式：**  
  * 57 (“2024年1-3月工业增加值”) 显示3月份数据（及一季度累计）于**2024年5月9日**发布。这对3月份数据而言滞后显著（约39天）。这似乎是一个详细数据表的发布。  
  * 19/21/22 (“广东2025年一季度GDP增长4.1%……规模以上工业增加值增长3.9%……3月份显著增长5.5%”) 这篇日期为**2025年4月22/23日**的新闻报道讨论了一季度数据，包括3月份的月度工业产出数据。这表明3月份的工业增加值在4月22日前已知。3月份数据的滞后：约22天。  
  * 58 (广州市，非省份，但具指示性) 1-2月数据随新闻报道于**2024年3月26日**发布。  
  * 20 (“广东2024年前四个月工业增长7.0%”) 新闻报道日期为**2024年5月21日**，暗示4月份数据届时已可用。4月份数据的滞后：约21天。  
  * 23 建议发布日期在每月22日左右。23 也暗示在月末/下月初。23 和 23 确认2025年第一季度工业增加值于2025年4月22日发布。2025年1-2月数据于2025年4月3日发布。  
  * 模式似乎是月度工业增加值数据通常作为更广泛的“月度经济运行报告”或包含月度细分的季度总结的一部分发布。  
* **支持材料：** 19 (市级), 20。  
* **滞后计算：** 基于4月22/23日发布3月数据，5月21日发布4月数据，约22-25天的滞后似乎常见。  
* **建议滞后：** **参考月末后25个日历日**。这考虑了数据作为更广泛经济报告一部分发布的情况。  
  广东省的月度工业增加值数据似乎比国家统计局的IVA数据发布得晚，并且通常作为更广泛经济更新的一部分，而非独立的工业报告。国家统计局的IVA数据稳定在月中发布（约18天滞后）1。相比之下，广东3月份的IVA数据到4月22日才可用（约22天滞后）19，4月份的IVA数据到5月21日可用（约21天滞后）20。57显示，3月份IVA的详细表格甚至更晚，在5月9日才发布。这表明存在一个层级：国家数据先行，省级细分数据随后，且常被整合到综合经济报告中 19。由于缺乏像国家统计局那样的固定省级日历 1，确定确切日期更为困难，需要依赖新闻报道日期或省级统计网站的“最新发布”栏目 23。模型必须考虑到省级IVA相对于国家IVA大约额外一周的滞后。这种可变性也意味着，对于回溯测试而言，采用稍保守的滞后（例如25天）更为安全。数据可能先出现在新闻摘要中，之后才会发布详细的统计表格。

**2\. 01\_广东\_用电量\_当月值**

* **发布机构：** 可能为广东省统计局、广东省发展和改革委员会 (DRC) / 能源局，或中国南方电网。关键是找到*最早、公开、官方*的来源。  
* **发布模式：**  
  * 27：“广东2024年全社会用电量达到9121亿千瓦时”。此新闻日期为**2025年1月22日**，指的是2024年*年度*数据。  
  * 28：“广东上半年全社会用电量达到4134亿千瓦时”。新闻日期为**2024年8月13日**，针对2024年上半年数据。这对月度成分而言滞后显著。  
  * 59：“广东电力发展股份有限公司”的半年报摘要提及“广东省2024年1-6月全社会用电量为4134.2亿千瓦时，同比增长8.5%”。公告日期为2024-48（可能是8月/9月发布上半年数据）。这是公司报告，不一定是官方省级总量。  
  * 24/24：2020年3月的新闻提及南方电网数据和广州供电局数据。这表明电网公司是一个来源，但*官方省级总量*的发布日程不明确。  
  * 60 显示CEIC从NBS报告广东用电量数据，但为*年度*数据。  
  * 61 表明在材料中难以直接从省级政府机构找到清晰、常规的月度发布信息。64 指向广东省能源局（隶属发改委）网站。  
* **支持材料：** 27 (年度), 28 (上半年), 59 (上半年, 公司), 24 (电网公司, 历史), 60 (NBS经CEIC, 年度)。  
* **滞后计算：** 材料中缺乏直接的月度发布信息。省级电力数据相对于IVA或GDP等核心经济指标，其常规、及时的月度发布更难确定。它可能被汇编并以更长的滞后时间发布，或者作为不那么频繁的更广泛能源报告的一部分。  
* **建议滞后：** 鉴于材料中缺乏*官方省级总用电量*的明确月度发布证据，此项较难确定。如果无法确定具有明确滞后性的持续月度官方来源，则可能需要通过国家趋势或行业特定报告进行代理，或者必须假定更保守的滞后。然而，如果我们假设它可能是不太频繁的能源特定报告或比IVA更晚发布的更广泛经济摘要的一部分：**参考月末后45个日历日**。这是一个估计值，如果找到更精确的地方发布信息，应予以验证。  
  在所提供的材料中，持续、及时的*官方省级总用电量*数据不像IVA或GDP等其他指标那样容易获得或有明确的发布计划。电网公司（例如，广东的中国南方电网 24）拥有这些数据，但可能不会按固定、频繁的时间表公开发布官方省级汇总数据。国家能源局的全国总用电量数据滞后约20天。广东IVA滞后约25天。广东电力相关的材料 27 显示的是年度或半年度数据发布，这意味着月度数据要么不定期发布，要么深嵌在不那么频繁的报告中。像24这样的新闻报道提到了电网公司的数据，但这并不总是由统计局或能源局发布的官方、汇总的省级总量。在材料中，未能从site:drc.gd.gov.cn "能源" "月度" "电力消费" OR "用电量" "2023" 29 或特定的能源局月度报告中找到相关信息，这一点颇能说明问题。用户可能在获取广东及时的历史月度用电量方面面临挑战。如果官方数据滞后严重（例如，超过30-40天），其对高频模型的效用会降低。他们可能需要：a. 接受更长的滞后（例如，45天作为占位符）。b. 调查广东电网或省能源局是否有不那么明显但定期的数字发布点。c. 如果省级数据过于滞后或无法获得，考虑使用国家用电量变化作为代理。d. 对于回溯测试，如果能找到历史数据点的实际发布日期（即使不规律），则应使用这些日期。

### **B. 湖北省**

**1\. 01\_湖北\_工业增加值\_可比价\_规模以上工业企业\_当月同比**

* **发布机构：** 湖北省统计局。  
* **发布模式：**  
  * 66：武汉市（市级）一季度高技术制造业IVA于4月29/30日发布。省级数据通常遵循类似模式，主要数据可能稍早或同时发布。  
  * 67 (CEIC) 表明湖北VAI（工业增加值）同比年初至今（实际值）每月更新，2025年2月数据可用。这暗示了月度发布。CEIC注明数据从2000年1月至2025年2月。  
  * 68 (IMF SDDS关于中国工业生产指数，IVA是其中一部分)：“新闻稿在中国国家统计局官方网站上于参考月份结束后20天内发布”。这指的是国家数据。省级数据通常较晚。  
  * 35：这些浏览摘要常表明，从湖北统计局网站主页或初步搜索中难以找到*典型发布日期*，暗示发布可能在报告内部，而非突出的独立数据。35提及2025年一季度工业经济增长解读于2025年4月23日发布。  
  * 查看湖北实际经济报告：69 (湖北省2024年国民经济和社会发展统计公报，2025年3月21日发布) 给出2024年*年度*IVA。70 (湖北省2024年经济运行新闻发布会，2025年1月22日) 也讨论了2024年度IVA。这些对于月度建模而言滞后过长。  
  * 71 (武汉市2024年一季度数据，2024年4月22日发布) 包括一季度工业增长并提及月度加速。这暗示在准备一季度总结时已知月度数据。  
* **支持材料：** 66 (市级), 67 (CEIC), 68 (国家背景), 71 (市级Q1，暗示月度), 35。  
* **滞后计算：** 与广东类似，可能作为更广泛经济报告的一部分发布，滞后于国家发布。如果一季度数据（包括3月月度）在4月22-23日左右讨论，这对3月数据而言是约22-23天的滞后。  
* **建议滞后：** **参考月末后25个日历日**。（与广东一致，假设在材料中缺乏湖北特定发布日历的情况下，省级报告实践相似）。

**2\. 01\_湖北\_用电量\_当月值**

* **发布机构：** 湖北省发展和改革委员会 (DRC) / 能源局，或国家电网湖北公司。  
* **发布模式：**  
  * 30：“2024年1月全省发用电情况”由“电力调度处”（可能隶属于发改委或国家电网）于**2024年2月29日**发布。1月数据滞后约29天。  
  * 31/32：“2024年3月全省发用电情况”由“电力调度处”于**2024年4月11日**发布。3月数据滞后约11天。  
  * 33：“2024年4月全省发用电情况”由“电力调度处”于**2024年5月10日**发布。4月数据滞后约10天。  
  * 这显示了“电力调度处”在M+1月10-11日左右非常一致和及时的发布。这远好于为广东推断的情况。  
* **支持材料：** 5 (湖北发改委电力数据的一般性提及), 30。61 和 61 也将此数据指向湖北发改委。  
* **滞后计算：** 根据31\-33，滞后稳定在月末后10-11天。  
* **建议滞后：** **参考月末后12个日历日**。这特定于湖北，并基于其电力调度处及时发布的有力证据。  
  即使是相似的指标，省际间数据的及时性和清晰度也可能存在显著差异。湖北省的月度用电量数据，来源于其“电力调度处”（可能隶属于省发改委或国家电网），发布相当及时（约10-12天滞后）30。这与在所提供材料中为广东省总用电量寻找类似常规、及时来源时遇到的明显困难形成对比。国家能源局的全国用电量数据滞后约20天。湖北的这种及时性表明，投资模型需要针对此类省级差异进行定制。如果这种模式持续，湖北的电力数据可能比其他数据发布更滞后或透明度较低的省份的区域经济指标更为及时。这也凸显了深入研究特定省级机构网站（如湖北发改委电力调度处页面）而非仅仅是主要统计局页面的重要性。

**表3：省级层面月度指标 \- 发布滞后性**

| 指标名称 | 省份 | 发布机构 | 典型发布模式 | 建议滞后 (月末后台历日) | 支持材料 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 工业增加值 (规模以上) | 广东 | 广东省统计局 | M+1月22-25日左右 | 25 | 19 |
| 全社会用电量 | 广东 | 省能源相关部门/电网 | 不明确，可能M+1月下旬至M+2月 | 45 (估计值) | 27 (年度/半年度) |
| 工业增加值 (规模以上) | 湖北 | 湖北省统计局 | M+1月22-25日左右 | 25 | 66 (市级), 67 (CEIC) |
| 全社会用电量 | 湖北 | 省发改委电力调度处 | M+1月10-12日左右 | 12 | 30\-33 |

## **IV. 季度国内生产总值 (GDP)：发布滞后性分析**

GDP数据是每季度发布的基石性宏观经济指标。

### **A. 01\_中国\_GDP\_现价\_累计值**

* **发布机构：** 国家统计局 (NBS)。  
* **发布模式：** 季度GDP作为“国民经济运行情况”新闻发布会的一部分发布。1 (第1行) 显示了这些日期：  
  * Q1 (1-3月): 4月16-18日 (例如，2024年为4月16日/周三 1; 2025年Q1为4月18日 2)  
  * Q2 (1-6月): 7月15-16日  
  * Q3 (1-9月): 10月20-21日  
  * Q4 (全年): 次年1月17-19日 通常发布的是年初至今的累计GDP、该累计数字的同比增长率，以及特定季度的同比增长率。  
* **支持材料：** 1 (第1行), 2 (“2025年第一季度GDP初步核算结果”于4月18日发布), 34 (指出季度GDP通常在季后15日左右发布，但官方日历更为精确)。  
* **滞后计算：**  
  * Q1 (3月31日结束): 约4月16-18日发布。滞后约16-18天。  
  * Q2 (6月30日结束): 约7月15-16日发布。滞后约15-16天。  
  * Q3 (9月30日结束): 约10月20-21日发布。滞后约20-21天。  
  * Q4 (12月31日结束): 约1月17-19日发布。滞后约17-19天。  
* **建议滞后：** **参考季度末后20个日历日**。这为所有季度提供了一个统一的缓冲。  
  最初的季度GDP发布（约20天滞后）是“初步核算数” 34。这些数据可能会有修订。34还提到“年度GDP最终核实数于隔年的1月份”发布。虽然用户主要关心的是用于建模的首次发布滞后，但重要的是要认识到这些数字可能会发生变化。重大修订，特别是“最终核实”，会晚得多，并将构成一个独立的信息事件。对于基于初步市场反应的交易模型，使用初步数据并应用约20天的滞后是正确的。然而，对于更长期的经济分析或可能对数据修订敏感的模型，了解修订周期非常重要。用户应使用约20天后可获得的初步数据。

### **B. 01\_广东\_GDP\_累计值**

* **发布机构：** 广东省统计局。  
* **发布模式：** 省级GDP数据通常在国家数据发布后几天到一周内发布。  
  * 2024年Q1：国家GDP可能在4月16-18日发布。72显示广东2024年Q1 GDP在一份日期为**2024年4月22日**的报告中发布。滞后约22天。  
  * 2024年H1 (Q2)：73 (7月23/24日新闻报道，数据来自广东省统计局) 和 74 (7月30日发布的数据表) 表明发布时间在7月23-30日之间。国家Q2 GDP在7月中旬发布。广东Q2滞后：约23-30天。  
  * 2024年Q1-Q3：75/76 (财新关于广东省统计局数据的报道) 指出数据于10月21/22日发布。国家Q3 GDP在10月20-21日左右。广东Q3滞后：约21-22天。  
  * 2025年Q1：19/21/22 新闻报道日期为2025年4月22/23日，针对Q1数据。滞后约22-23天。  
  * 23 (浏览摘要) 确认Q1在4月下旬，Q2在7月底，Q3在10月底。23和23证实了这些模式。  
* **支持材料：** 19。  
* **滞后计算：** 通常在季度结束后21-25天，比国家数据晚几天。  
* **建议滞后：** **参考季度末后25个日历日**。

### **C. 01\_湖北\_GDP\_累计值**

* **发布机构：** 湖北省统计局。  
* **发布模式：** 与广东类似，湖北的GDP预计在国家数据发布之后。  
  * 2024年Q1：71 (武汉市Q1 GDP，2024年4月22日发布)。湖北省Q1 GDP可能在相近时间或稍早发布，因为市级数据通常跟随省级。  
  * 2024年H1 (Q2)：77 (湖北H1 GDP新闻报道，2024年7月20日)。滞后约20天。78 (武汉H1 GDP新闻报道，2024年7月23日)。  
  * 2024年Q1-Q3：79/80 (湖北Q1-Q3 GDP新闻报道，2024年10月22/23日)。滞后约22-23天。  
  * 2025年Q1：35提及湖北省2025年Q1新闻发布会于2025年4月18日举行。这非常及时，与国家发布时间相同。  
  * 35 (浏览摘要) 指出湖北Q1 GDP在4月下旬左右，但35显示的2025年Q1于2025年4月18日发布则更快。这表明及时性可能有所改善或存在变动。  
* **支持材料：** 81 (年度), 71 (市级Q1), 77 (市级H1), 35。  
* **滞后计算：** 湖北省似乎在季度结束后相当迅速地发布其GDP数据，有时与国家发布时间一致或非常接近。2025年Q1为4月18日。2024年H1 (Q2)为7月20日。2024年Q1-Q3为10月22/23日。这是一个18-23天的范围。  
* **建议滞后：** **参考季度末后23个日历日**。这考虑了微小变动，并与观察到的发布时间较晚的一端对齐。  
  一些省份可能力求其关键经济数据（如GDP）的发布时间非常接近甚至与国家统计局的发布同步。湖北省2025年第一季度GDP新闻发布会于4月18日举行 35，与国家统计局2025年第一季度GDP的发布日期（4月18日）2 一致，这便是一个例证。这可能旨在显示经济实力或效率。国家2025年Q1 GDP于4月18日发布 2。湖北于4月18日举行了其2025年Q1经济运行新闻发布会（推测包含GDP）35。广东2025年Q1 GDP约在4月22/23日报道 19。这表明就2025年Q1而言，湖北比广东快，并与国家发布同步。然而，对于2024年的其他季度（Q2, Q3），湖北的滞后略长（20-23天）。因此，不存在单一的“省级GDP滞后”。它可能因省而异，甚至同一省份的不同季度也可能不同。对于建模，除非对每个历史数据点的特定发布日期进行细致跟踪，否则使用一致但略保守的滞后（例如省级23-25天）是一种实用方法。如果一个省份持续与国家在同一天发布，则可对该特定省份使用国家滞后（例如20天）。

**表4：季度GDP指标 \- 发布滞后性**

| 指标名称 | 地区 | 发布机构 | 典型发布模式 | 建议滞后 (季末后台历日) | 支持材料 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| GDP累计值 | 中国 | NBS | Q+1月16-21日左右 | 20 | 1 |
| GDP累计值 | 广东 | 广东省统计局 | Q+1月21-30日左右 | 25 | 19 |
| GDP累计值 | 湖北 | 湖北省统计局 | Q+1月18-23日左右 | 23 | 35 (for Q1'25) |

## **V. 日度频率数据 (D类)：对齐说明**

用户查询列出了几项日度数据系列：欧洲ARA港动力煤现货价、布伦特原油期货结算价、欧盟排放配额(EUA)期货结算价、NYMEX天然气期货收盘价，以及CFETS美元兑人民币即期汇率。

这些是市场交易工具或官方汇率。

* 期货（布伦特、EUA、NYMEX天然气）：结算价通常在交易日（T日）结束时（EOD）可用 36。  
* 现货价格（ARA煤炭）：每日现货评估价通常也在EOD或T+1日可用。Argus/McCloskey的API 2（ARA煤炭）每日发布 39。  
* 汇率（CFETS美元兑人民币）：CFETS中间价每日早晨设定。即期汇率在整个交易日内均可获得，官方收盘即期汇率由CFETS发布。82显示“即期USD/CNY收盘跌71个基点报7.2083”，并附有时间戳。83显示“USD/CNY即期收盘价(16:30收盘价)”当日可用。

**建议对齐方式：**

* 对于期货结算价和现货商品评估价：使用T日的价格作为T+1日建模时可用的数据。实际上是**1天的滞后**，以确保数据在市场收盘后被捕获和传播。  
* 对于CFETS美元兑人民币：北京时间16:30的收盘即期汇率当日可用。出于建模目的，这可以被视为在T日EOD可用，因此根据模型执行时间，滞后为**0到1天**。如果模型在夜间或次日早晨运行，则T日的EOD汇率是合适的。

用户计划对周末的日度数据进行前向填充，这是与通常代表整个时期的低频（月度/季度）数据对齐的标准做法。这确保了当一个月度指标（例如1月份CPI）从其2月份的发布日期开始“应用”，直到下一个CPI发布之前，不会因未来的每日价格变动而产生前视偏差。

## **VI. 在日度投资模型中应用发布滞后性的建议**

### **A. 建议滞后性汇总表**

此表是为用户准备的主要交付成果，汇总了所有M类和Q类数据的建议滞后。

**表5：日度模型对齐的建议发布滞后性汇总**

| 数据系列 (文件名) | 地区 | 频率 | 发布机构 | 典型发布日/窗口 (例如：M+1月15-17日, M月末, Q+18-20日) | 建议滞后 (周期结束后台历日) | 备注/主要证据材料 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 01\_广东\_工业增加值\_可比价\_规模以上工业企业\_当月同比 | 广东 | M | 广东省统计局 | M+1月22-25日左右 | 25 | 19 |
| 01\_湖北\_工业增加值\_可比价\_规模以上工业企业\_当月同比 | 湖北 | M | 湖北省统计局 | M+1月22-25日左右 | 25 | 66 (市级), 67 (CEIC) |
| 01\_中国\_产量\_原油加工量\_当月值 | 中国 | M | NBS | M+1月14-19日 | 18 | 1 |
| 01\_中国\_产量\_原煤\_当月值 | 中国 | M | NBS | M+1月14-19日 | 18 | 1 |
| 01\_中国\_产量\_水泥\_当月值 | 中国 | M | NBS | M+1月14-19日 | 18 | 1 |
| 01\_中国\_产量\_粗钢\_当月值 | 中国 | M | NBS | M+1月14-19日 | 18 | 1 |
| 01\_中国\_发电量\_火电\_当月值 | 中国 | M | NBS | M+1月14-19日 | 18 | 1 |
| 01\_中国\_全社会用电量\_当月值 | 中国 | M | NEA | M+1月17-20日 | 20 | 7\-11 |
| 01\_中国\_社会融资规模\_当月值 | 中国 | M | PBOC | M+1月9-15日 | 16 | 18 |
| 01\_广东\_用电量\_当月值 | 广东 | M | 省能源部门/电网 | M+1月下旬至M+2月 (不明确) | 45 (估计值) | 27 |
| 01\_湖北\_用电量\_当月值 | 湖北 | M | 省发改委电力调度处 | M+1月10-12日 | 12 | 30\-33 |
| 01\_中国\_CPI\_当月同比 | 中国 | M | NBS | M+1月9-10日 (10月约15日) | 11 | 1 |
| 01\_中国\_制造业PMI | 中国 | M | NBS | M月末或M+1月1日 | 2 | 1 |
| 01\_中国\_GDP\_现价\_累计值 | 中国 | Q | NBS | Q+1月16-21日左右 | 20 | 1 |
| 01\_广东\_GDP\_累计值 | 广东 | Q | 广东省统计局 | Q+1月21-30日左右 | 25 | 72 |
| 01\_湖北\_GDP\_累计值 | 湖北 | Q | 湖北省统计局 | Q+1月18-23日左右 | 23 | 35 |

### **B. 关于合并数据发布（如1-2月数据）的处理指南**

由于春节假期影响，国家统计局通常在3月中旬合并发布1月和2月的经济数据（如工业增加值、工业产出、零售额、固定资产投资）1。例如，“国民经济运行情况”在1月17日发布后，通常会跳过2月，然后在3月17日发布，这暗示1-2月数据包含在3月的发布中。

当1-2月合并数据发布时（例如3月15日），这是1月和2月实际数据首次为公众所知的时间点。

* 对于1月数据：滞后时间从1月31日至约3月15日（约43-45天）。  
* 对于2月数据：滞后时间从2月28/29日至约3月15日（约15-16天）。

**建议建模方法：** 模型应反映1月和2月的值在3月的发布日*同时*变得可用。日度序列应从该日期开始更新1月的值，并从同一日期开始更新2月的值，实际上在3月中旬之前，1月的数据会保持一个“占位符”（例如，上个月的值或预测值）。

春节对数据报告的合并发布是中国经济数据的一个显著结构性特征。这不仅仅是数据延迟，它反映了春节这一重大经济社会事件的影响。春节时间可能在1月或2月，使得单月同比比较产生误导。合并发布为年初提供了一个更稳定的同比比较基准。这意味着市场在近两个月内可能缺乏某些关键指标的最新数据。模型需要能够稳健地处理这种数据缺口。在此期间，预测或替代数据可能被更频繁地依赖。3月份的同时发布可能导致更大的市场调整，因为两个月的信息被同时消化。所建议的滞后应用确保了这种真实世界的数据流在模型中得到反映。

### **C. 关于数据初版和修订的考虑（简述）**

经济数据，特别是GDP，可能会有修订。本报告中建议的滞后性指的是数据的*首次*发布，因为这是市场参与者最初会做出反应的依据 34。对于基于初始市场反应的回溯测试交易策略，使用首次发布的数据并应用适当的滞后是正确的。如果模型旨在进行更长期的基本面分析，用户可能需要考虑如何纳入修订后的数据，但这是一个独立于初始发布滞后性的“数据版本”管理问题。建议使用首次发布的数据。如果使用的数据库提供修订后的历史序列，并且模型对此敏感，则应确保回溯测试能够访问特定时间点的数据版本。

### **D. 关于日度对齐前向填充的说明**

确认用户计划使用最后一个有效日度值对周末的日度序列进行前向填充。当新的月度或季度数据点发布（在应用其发布滞后之后），代表该经济变量的日度序列应从计算出的发布日期的*次日*开始更新为新值。然后，这个新值将被前向填充，直到下个月/季度的数据变得可用。例如，如果1月份CPI（参考期结束：1月31日）有11天的滞后，它将在2月11日发布。模型中的日度CPI序列将从2月12日开始采用新的1月份CPI值，并保持该值直到3月份CPI发布并应用。

## **VII. 结论**

准确的发布滞后性在金融建模中起着关键作用。本报告中建议的滞后是基于官方来源和观察到的发布模式的最佳可用证据。对于持续的模型维护，建议监测这些机构的发布日程，因为实践可能会随时间演变。特别是省级数据的发布，其及时性和透明度可能因省而异，需要模型开发者进行细致的个案研究和持续关注。

## **VIII. 附录 (可选)**

主要国家级和省级统计/能源机构网站（部分已在正文引用中提及）：

* 国家统计局 (NBS): www.stats.gov.cn  
* 中国人民银行 (PBOC): www.pbc.gov.cn  
* 国家能源局 (NEA): www.nea.gov.cn  
* 广东省统计局: stats.gd.gov.cn  
* 湖北省统计局: tjj.hubei.gov.cn  
* 广东省发展和改革委员会 (能源局): drc.gd.gov.cn (能源局通常隶属于发改委)  
* 湖北省发展和改革委员会: fgw.hubei.gov.cn

#### **Works cited**

1. 最新统计信息发布日程- 国家统计局, accessed May 17, 2025, [https://www.stats.gov.cn/sj/fbrc/](https://www.stats.gov.cn/sj/fbrc/)  
2. National Bureau of Statistics of China \>\> Latest Releases, accessed May 17, 2025, [https://www.stats.gov.cn/english/PressRelease/](https://www.stats.gov.cn/english/PressRelease/)  
3. Consumption, Investment, Industry and Housing Market | China Monthly Economic Data | Collection | MacroMicro, accessed May 17, 2025, [https://en.macromicro.me/collections/21031/china-monthly-economic-data](https://en.macromicro.me/collections/21031/china-monthly-economic-data)  
4. 2024年全社会用电量同比增长6.8% \- 国家能源局, accessed May 17, 2025, [https://www.nea.gov.cn/20250120/4f7f249bac714e7693adecac996d742f/c.html](https://www.nea.gov.cn/20250120/4f7f249bac714e7693adecac996d742f/c.html)  
5. 2024年全社会用电量同比增长6.8% \- 湖北省发展和改革委员会, accessed May 17, 2025, [https://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/nyfzzlc/gzdt/202502/t20250212\_5538250.shtml](https://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/nyfzzlc/gzdt/202502/t20250212_5538250.shtml)  
6. 2024年全社会用电量同比增长6.8% \- 福建省工业和信息化厅, accessed May 17, 2025, [https://gxt.fujian.gov.cn/zwgk/xw/hydt/xydt/202501/t20250121\_6705420.htm](https://gxt.fujian.gov.cn/zwgk/xw/hydt/xydt/202501/t20250121_6705420.htm)  
7. 国家能源局：2月份全社会用电量同比增长8.6% \- 新闻频道- 央视网, accessed May 17, 2025, [https://news.cctv.com/2025/03/18/ARTIBGxZimVRdPm2cEh7ENNG250318.shtml](https://news.cctv.com/2025/03/18/ARTIBGxZimVRdPm2cEh7ENNG250318.shtml)  
8. 2025年2月份全社会用电量同比增长8.6% \- 广东宝丽华新能源股份有限公司, accessed May 17, 2025, [http://www.baolihua.com.cn/news/industry\_1300.html](http://www.baolihua.com.cn/news/industry_1300.html)  
9. 3月份全社会用电量同比增长7.4% \- 中国政府网, accessed May 17, 2025, [https://www.gov.cn/lianbo/bumen/202404/content\_6945861.htm](https://www.gov.cn/lianbo/bumen/202404/content_6945861.htm)  
10. 2024年1-3月份国内能源信息 \- 吉林省能源局, accessed May 17, 2025, [http://nyj.jl.gov.cn/xwdt/nyyw/202404/t20240424\_3149203.html](http://nyj.jl.gov.cn/xwdt/nyyw/202404/t20240424_3149203.html)  
11. 4月份全社会用电量同比增长7.0% \- 国家能源局, accessed May 17, 2025, [https://www.nea.gov.cn/2024-05/17/c\_1212363227.htm](https://www.nea.gov.cn/2024-05/17/c_1212363227.htm)  
12. 国家能源局：2025年3月份全社会用电量同比增长4.8% \- 新华网, accessed May 17, 2025, [http://www3.xinhuanet.com/politics/20250418/6fa2b31deda146b5bae305cd0fbba64c/c.html](http://www3.xinhuanet.com/politics/20250418/6fa2b31deda146b5bae305cd0fbba64c/c.html)  
13. 2025年3月份全社会用电量同比增长4.8% \- 国家能源局, accessed May 17, 2025, [https://www.nea.gov.cn/20250418/b713f0ff6b314575923add3154070fe8/c.html](https://www.nea.gov.cn/20250418/b713f0ff6b314575923add3154070fe8/c.html)  
14. 2023年全社会用电量92241亿千瓦时同比增长6.7% \- 国家能源局, accessed May 17, 2025, [https://www.nea.gov.cn/2024-01/26/c\_1310762222.htm](https://www.nea.gov.cn/2024-01/26/c_1310762222.htm)  
15. 1至2月全社会用电量超1.53万亿千瓦时 \- 中国政府网, accessed May 17, 2025, [https://www.gov.cn/lianbo/bumen/202403/content\_6940513.htm](https://www.gov.cn/lianbo/bumen/202403/content_6940513.htm)  
16. 一季度全社会用电量同比增2.5% 用电结构优化 \- 中国政府网, accessed May 17, 2025, [https://www.gov.cn/lianbo/bumen/202504/content\_7019825.htm](https://www.gov.cn/lianbo/bumen/202504/content_7019825.htm)  
17. 中国-财新制造业采购经理人指数\[PMI\] | 数据| MacroMicro 财经M平方, accessed May 17, 2025, [https://sc.macromicro.me/series/948/cn-china-caixin-manufacturing-pmi](https://sc.macromicro.me/series/948/cn-china-caixin-manufacturing-pmi)  
18. Financial Statistics Report (April 2025), accessed May 17, 2025, [http://www.pbc.gov.cn/en/3688247/3688978/3709137/5711424/index.html](http://www.pbc.gov.cn/en/3688247/3688978/3709137/5711424/index.html)  
19. Guangdong's GDP grows 4.1% in Q1 2025 as manufacturing and high-tech industries drive growth \- Guangzhou, accessed May 17, 2025, [http://www.eguangzhou.gov.cn/gzbusiness/content/post\_33526.html](http://www.eguangzhou.gov.cn/gzbusiness/content/post_33526.html)  
20. Guangdong shows 7.0% industrial surge in first four months of 2024, reflecting a stable and evolving economy \- Newsgd.com, accessed May 17, 2025, [https://www.newsgd.com/node\_a21acd2229/d0b91da42a.shtml](https://www.newsgd.com/node_a21acd2229/d0b91da42a.shtml)  
21. Guangdong's GDP grows 4.1% in Q1 2025 as manufacturing and high-tech industries drive growth \- Guangzhou, accessed May 17, 2025, [http://www.eguangzhou.gov.cn/gzlatest/content/post\_33525.html](http://www.eguangzhou.gov.cn/gzlatest/content/post_33525.html)  
22. GDP growth of 4.1%, Guangdong's economic data for the first quarter released, accessed May 17, 2025, [https://www.lwxsd.com/en/info\_view.php?tab=mynews\&VID=64964](https://www.lwxsd.com/en/info_view.php?tab=mynews&VID=64964)  
23. 欢迎光临广东统计信息网, accessed May 17, 2025, [http://stats.gd.gov.cn/](http://stats.gd.gov.cn/)  
24. The resumption of work and production is accelerating, and Guangdong's electricity consumption is growing strongly \- EEWORLD, accessed May 17, 2025, [https://en.eeworld.com.cn/news/newenergy/eic609333.html](https://en.eeworld.com.cn/news/newenergy/eic609333.html)  
25. CSG Accelerates Construction of Digital Power Grids, accessed May 17, 2025, [http://en.ccceu.eu/2024-04/30/c\_4240.htm](http://en.ccceu.eu/2024-04/30/c_4240.htm)  
26. China Southern Power Grid | World Economic Forum, accessed May 17, 2025, [https://www.weforum.org/organizations/china-southern-power-grid-co/](https://www.weforum.org/organizations/china-southern-power-grid-co/)  
27. 全国首个！广东2024年全社会用电量突破9000亿千瓦时 \- 中国能源新闻网, accessed May 17, 2025, [https://www.cpnn.com.cn/news/dfny/202501/t20250122\_1768803.html](https://www.cpnn.com.cn/news/dfny/202501/t20250122_1768803.html)  
28. 广东上半年全社会用电量位居全国首位, accessed May 17, 2025, [http://www.sasac.gov.cn/n2588025/n2588124/c31411288/content.html](http://www.sasac.gov.cn/n2588025/n2588124/c31411288/content.html)  
29. 广东省能源局国家能源局南方监管局关于2025年电力市场交易有关事项的通知, accessed May 17, 2025, [http://drc.gd.gov.cn/snyj/tzgg/content/post\_4582745.html](http://drc.gd.gov.cn/snyj/tzgg/content/post_4582745.html)  
30. 2024年1月全省发用电情况 \- 湖北省发展和改革委员会, accessed May 17, 2025, [http://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/dlddc/tzgg/202402/t20240229\_5101108.shtml](http://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/dlddc/tzgg/202402/t20240229_5101108.shtml)  
31. 2024年3月全省发用电情况 \- 湖北省发展和改革委员会, accessed May 17, 2025, [https://fgw.hubei.gov.cn/fgjj/sjsfg/sjfx/lx/nyjs/202404/t20240411\_5156298.shtml](https://fgw.hubei.gov.cn/fgjj/sjsfg/sjfx/lx/nyjs/202404/t20240411_5156298.shtml)  
32. 2024年3月全省发用电情况, accessed May 17, 2025, [http://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/dlddc/tzgg/202404/t20240411\_5156298.shtml](http://fgw.hubei.gov.cn/fbjd/xxgkml/jgzn/wgdw/nyj/dlddc/tzgg/202404/t20240411_5156298.shtml)  
33. 2024年4月全省发用电情况 \- 湖北省发展和改革委员会, accessed May 17, 2025, [https://fgw.hubei.gov.cn/fgjj/sjsfg/sjfx/lx/nyjs/202405/t20240510\_5186514.shtml](https://fgw.hubei.gov.cn/fgjj/sjsfg/sjfx/lx/nyjs/202405/t20240510_5186514.shtml)  
34. 我国如何核算季度GDP？\_重庆市黔江区人民政府, accessed May 17, 2025, [https://www.qianjiang.gov.cn/bmjd/xzfgzbm/qtjj/zwgk\_49283/gkml/zczxk/202401/t20240126\_12869663.html](https://www.qianjiang.gov.cn/bmjd/xzfgzbm/qtjj/zwgk_49283/gkml/zczxk/202401/t20240126_12869663.html)  
35. 湖北省统计局, accessed May 17, 2025, [http://tjj.hubei.gov.cn/](http://tjj.hubei.gov.cn/)  
36. Light Sweet Crude Oil (CL) Futures Daily Settlement Procedure \- CME Group Client Systems Wiki \- Confluence, accessed May 17, 2025, [https://cmegroupclientsite.atlassian.net/wiki/display/epicsandbox/nymex+crude+oil](https://cmegroupclientsite.atlassian.net/wiki/display/epicsandbox/nymex+crude+oil)  
37. 'ACCC review of the LNG netback price series' – CME Group Submission, accessed May 17, 2025, [https://www.accc.gov.au/system/files/CME%20Group%20Submission-%20LNG%20netback%20reivew\_Redacted.pdf](https://www.accc.gov.au/system/files/CME%20Group%20Submission-%20LNG%20netback%20reivew_Redacted.pdf)  
38. EUA Futures \- ICE, accessed May 17, 2025, [https://www.ice.com/products/197/eua-futures](https://www.ice.com/products/197/eua-futures)  
39. Coal \- Price \- Chart \- Historical Data \- News \- Trading Economics, accessed May 17, 2025, [https://tradingeconomics.com/commodity/coal](https://tradingeconomics.com/commodity/coal)  
40. World coal market: brief overview, accessed May 17, 2025, [https://thecoalhub.com/world-coal-market-brief-overview-160.html](https://thecoalhub.com/world-coal-market-brief-overview-160.html)  
41. Argus/McCloskey's Coal Price Index Report, accessed May 17, 2025, [https://www.argusmedia.com/en/solutions/products/argus-mccloskeys-coal-price-index-service](https://www.argusmedia.com/en/solutions/products/argus-mccloskeys-coal-price-index-service)  
42. Argus/McCloskey's API 2 \- Coal price index, accessed May 17, 2025, [https://www.argusmedia.com/en/methodology/key-commodity-prices/argus-mccloskeys-api-2](https://www.argusmedia.com/en/methodology/key-commodity-prices/argus-mccloskeys-api-2)  
43. China's major economic data continue to improve | PIIE, accessed May 17, 2025, [https://www.piie.com/blogs/realtime-economics/2025/chinas-major-economic-data-continue-improve](https://www.piie.com/blogs/realtime-economics/2025/chinas-major-economic-data-continue-improve)  
44. Corporate Calendar \- Online Banking, accessed May 17, 2025, [https://www.boc.cn/EN/INVESTOR/ir8/200810/t20081024\_7837.html](https://www.boc.cn/EN/INVESTOR/ir8/200810/t20081024_7837.html)  
45. China Calendar \- Trading Economics, accessed May 17, 2025, [https://tradingeconomics.com/china/calendar](https://tradingeconomics.com/china/calendar)  
46. 原油月报Crude Oil Monthly Report, accessed May 17, 2025, [https://pdf.dfcfw.com/pdf/H3\_AP202409191639929094\_1.pdf](https://pdf.dfcfw.com/pdf/H3_AP202409191639929094_1.pdf)  
47. 5 月原煤产量恢复仍不理想, accessed May 17, 2025, [https://pdf.dfcfw.com/pdf/H3\_AP202406241636819457\_1.pdf](https://pdf.dfcfw.com/pdf/H3_AP202406241636819457_1.pdf)  
48. 中国交建2023年度ESG报告（定稿）, accessed May 17, 2025, [https://www.ccccltd.cn/shzr/shzr\_esgzq/esgzq\_esgbg/202410/P020241012473250068661.pdf](https://www.ccccltd.cn/shzr/shzr_esgzq/esgzq_esgbg/202410/P020241012473250068661.pdf)  
49. 2025年3月全球粗钢产量- worldsteel.org, accessed May 17, 2025, [https://worldsteel.org/zh-hans/media/press-releases/2025/march-2025-crude-steel-production/](https://worldsteel.org/zh-hans/media/press-releases/2025/march-2025-crude-steel-production/)  
50. 国家能源局2011年政府信息公开工作年度报告, accessed May 17, 2025, [http://zfxxgk.nea.gov.cn/ndbg/201203/t20120316\_1455.htm](http://zfxxgk.nea.gov.cn/ndbg/201203/t20120316_1455.htm)  
51. 全社会用电量上半年同比增长8.1% \- 人民日报, accessed May 17, 2025, [http://paper.people.com.cn/rmrbhwb/html/2024-07/24/content\_26070766.htm](http://paper.people.com.cn/rmrbhwb/html/2024-07/24/content_26070766.htm)  
52. 国家统计局, accessed May 17, 2025, [https://www.stats.gov.cn/](https://www.stats.gov.cn/)  
53. 居民消费价格指数 \- 北京市统计局, accessed May 17, 2025, [https://tjj.beijing.gov.cn/tjsj\_31433/yjdsj\_31440/cpi/2019/202002/P020200217556922481919.pdf](https://tjj.beijing.gov.cn/tjsj_31433/yjdsj_31440/cpi/2019/202002/P020200217556922481919.pdf)  
54. CPI同比 \- 国家统计局智能云搜索, accessed May 17, 2025, [https://www.stats.gov.cn/search/s?qt=CPI](https://www.stats.gov.cn/search/s?qt=CPI)  
55. PMI \- 国家统计局智能云搜索, accessed May 17, 2025, [https://www.stats.gov.cn/search/s?qt=PMI](https://www.stats.gov.cn/search/s?qt=PMI)  
56. 资管新规对中国影子银行监管的影响研究, accessed May 17, 2025, [https://www.hanspub.org/journal/paperinformation?paperid=68220](https://www.hanspub.org/journal/paperinformation?paperid=68220)  
57. 2024年1-3月工业增加值 \- 广东统计局, accessed May 17, 2025, [http://stats.gd.gov.cn/gyzjz/content/post\_4419568.html](http://stats.gd.gov.cn/gyzjz/content/post_4419568.html)  
58. 2024年1-2月广州经济运行情况, accessed May 17, 2025, [https://www.gz.gov.cn/zwgk/sjfb/tjfx/content/post\_9561638.html](https://www.gz.gov.cn/zwgk/sjfb/tjfx/content/post_9561638.html)  
59. Guangdong Electric power Development Co., Ltd. Summary of the Semi-Annual Report 2024, accessed May 17, 2025, [https://pdf.dfcfw.com/pdf/H2\_AN202408301639645418\_1.pdf](https://pdf.dfcfw.com/pdf/H2_AN202408301639645418_1.pdf)  
60. Guangdong: Electricity: Consumption | Economic Indicators \- CEIC, accessed May 17, 2025, [https://www.ceicdata.com/en/china/electricity-balance-sheet-guangdong/guangdong-electricity-consumption](https://www.ceicdata.com/en/china/electricity-balance-sheet-guangdong/guangdong-electricity-consumption)  
61. 湖北省发展和改革委员会, accessed May 17, 2025, [http://fgw.hubei.gov.cn/](http://fgw.hubei.gov.cn/)  
62. 广东省工业和信息化厅, accessed May 17, 2025, [http://www.gdei.gov.cn/](http://www.gdei.gov.cn/)  
63. 广东省生态环境厅, accessed May 17, 2025, [http://gdee.gd.gov.cn/](http://gdee.gd.gov.cn/)  
64. 广东省发展和改革委员会-, accessed May 17, 2025, [http://drc.gd.gov.cn/](http://drc.gd.gov.cn/)  
65. 广东省能源局 \- 广东省发展和改革委员会-, accessed May 17, 2025, [http://drc.gd.gov.cn/snyj/index.html](http://drc.gd.gov.cn/snyj/index.html)  
66. Wuhan's GDP up 5.4 pct in Q1 \- The people's government of hubei province, accessed May 17, 2025, [http://en.hubei.gov.cn/news/newslist/202504/t20250430\_5637401.shtml](http://en.hubei.gov.cn/news/newslist/202504/t20250430_5637401.shtml)  
67. VAI: YoY: Year to Date(Real): Hubei | Economic Indicators \- CEIC, accessed May 17, 2025, [https://www.ceicdata.com/en/china/value-added-of-industry-monthly/vai-yoy-ytdreal-hubei](https://www.ceicdata.com/en/china/value-added-of-industry-monthly/vai-yoy-ytdreal-hubei)  
68. SDDS \- DQAF View : China, People's Republic \- Production index \- Dissemination Standards Bulletin Board, accessed May 17, 2025, [https://dsbb.imf.org/sdds/dqaf-base/country/CHN/category/IND00](https://dsbb.imf.org/sdds/dqaf-base/country/CHN/category/IND00)  
69. 湖北省2024年国民经济和社会发展统计公报, accessed May 17, 2025, [https://tjj.hubei.gov.cn/tjsj/tjgb/ndtjgb/qstjgb/202503/t20250321\_5585085.shtml](https://tjj.hubei.gov.cn/tjsj/tjgb/ndtjgb/qstjgb/202503/t20250321_5585085.shtml)  
70. 2024年湖北经济运行情况新闻发布会, accessed May 17, 2025, [https://www.hubei.gov.cn/hbfb/xwfbh/202501/t20250122\_5515065.shtml](https://www.hubei.gov.cn/hbfb/xwfbh/202501/t20250122_5515065.shtml)  
71. 2024年一季度武汉市经济运行情况, accessed May 17, 2025, [https://tjj.wuhan.gov.cn/ztzl\_49/xwfbh/202404/t20240422\_2391544.shtml](https://tjj.wuhan.gov.cn/ztzl_49/xwfbh/202404/t20240422_2391544.shtml)  
72. 2024年一季度广东经济运行情况 \- 广东统计局, accessed May 17, 2025, [http://stats.gd.gov.cn/tjkx185/content/post\_4411614.html](http://stats.gd.gov.cn/tjkx185/content/post_4411614.html)  
73. 广东公布2024上半年经济数据：GDP超6.5万亿元, accessed May 17, 2025, [http://www.qb.gd.gov.cn/qwdt/content/post\_1253325.html](http://www.qb.gd.gov.cn/qwdt/content/post_1253325.html)  
74. 2024年上半年广东省地区生产总值, accessed May 17, 2025, [http://stats.gd.gov.cn/jdgnsczz/content/post\_4467630.html](http://stats.gd.gov.cn/jdgnsczz/content/post_4467630.html)  
75. economy.caixin.com, accessed May 17, 2025, [https://economy.caixin.com/2024-10-22/102248270.html\#:\~:text=%E3%80%90%E8%B4%A2%E6%96%B0%E7%BD%91%E3%80%912024%E5%B9%B4,%E8%AE%A1%E7%AE%97%EF%BC%8C%E5%90%8C%E6%AF%94%E5%A2%9E%E9%95%BF4.8%25%E3%80%82](https://economy.caixin.com/2024-10-22/102248270.html#:~:text=%E3%80%90%E8%B4%A2%E6%96%B0%E7%BD%91%E3%80%912024%E5%B9%B4,%E8%AE%A1%E7%AE%97%EF%BC%8C%E5%90%8C%E6%AF%94%E5%A2%9E%E9%95%BF4.8%25%E3%80%82)  
76. 房地产开发投资同比下降17.2% 前三季度广东GDP同比增长3.4% \- 经济, accessed May 17, 2025, [https://economy.caixin.com/2024-10-22/102248270.html](https://economy.caixin.com/2024-10-22/102248270.html)  
77. 进中提质！湖北上半年GDP增长5.8%，高技术制造业投资增8.9% \- 武汉市人民政府门户网站, accessed May 17, 2025, [https://www.wuhan.gov.cn/sy/whyw/202407/t20240720\_2431207.shtml](https://www.wuhan.gov.cn/sy/whyw/202407/t20240720_2431207.shtml)  
78. 刚刚，武汉2024年上半年GDP发布！ \- 湖北日报新闻客户端, accessed May 17, 2025, [https://news.hubeidaily.net/pc/c\_2900980.html](https://news.hubeidaily.net/pc/c_2900980.html)  
79. fgw.wuhan.gov.cn, accessed May 17, 2025, [https://fgw.wuhan.gov.cn/xwzx/fgyw/202410/t20241023\_2472552.html\#:\~:text=10%E6%9C%8822%E6%97%A5%EF%BC%8C%E7%9C%81,%E4%BA%8E%E5%85%A8%E5%9B%BD0.9%E4%B8%AA%E7%99%BE%E5%88%86%E7%82%B9%E3%80%82](https://fgw.wuhan.gov.cn/xwzx/fgyw/202410/t20241023_2472552.html#:~:text=10%E6%9C%8822%E6%97%A5%EF%BC%8C%E7%9C%81,%E4%BA%8E%E5%85%A8%E5%9B%BD0.9%E4%B8%AA%E7%99%BE%E5%88%86%E7%82%B9%E3%80%82)  
80. 湖北前三季度GDP同比增长5.7% 在经济大省中排名靠前, accessed May 17, 2025, [https://www.hubei.gov.cn/hbfb/bmdt/202410/t20241023\_5381690.shtml](https://www.hubei.gov.cn/hbfb/bmdt/202410/t20241023_5381690.shtml)  
81. 湖北GDP突破6万亿！背后是谁在发力？ \- 随县人民政府, accessed May 17, 2025, [http://www.zgsuixian.gov.cn/ztzl\_47/qszdtsg/202501/t20250123\_1297728.shtml](http://www.zgsuixian.gov.cn/ztzl_47/qszdtsg/202501/t20250123_1297728.shtml)  
82. Central Parity of USD/ RMB Adds 25 bps to 7.1938US Stocks \- Global News Content, accessed May 17, 2025, [http://freequote.aastocks.com/en/usq/news/comment.aspx?source=AAFN\&id=NOW.1440326\&catg=1](http://freequote.aastocks.com/en/usq/news/comment.aspx?source=AAFN&id=NOW.1440326&catg=1)  
83. CFETS \- China Foreign Exchange Trade System, accessed May 17, 2025, [https://www.chinamoney.org.cn/english/](https://www.chinamoney.org.cn/english/)