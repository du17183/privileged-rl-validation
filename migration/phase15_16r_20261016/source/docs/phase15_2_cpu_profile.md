# Phase15.2 独立 CPU Profiling

本报告由全部 288 个独立诊断 episode 的已关闭 collector 生成：A–E 同 modelseed1、同自然源字典序前32起点，共160；E四个 graded 源各前32，共128。只描述该独立 scope，不进入正式成功率、不采新自然或训练数据，不外推全部30500。正式300结果与冻结控制未改。

## 计时定义与分位数

wall、当前线程 CPU、子进程累计 CPU 分列；GPU forward/service/RPC 见独占延迟报告。inclusive span 包含内层 span，不能相加；exclusive 只在同一线程内相加。batch32 的共享动作/Isaac/RPC span 不除32伪造每episode实测。

cpu_context_timing.csv 保存每个原始 cohort/stage/thread 分层的 count、wall/thread CPU inclusive/exclusive P50/P95/P99。原 collector 未保存逐call样本，跨分层池化 P50/P95/P99 为 unknown；类别总数、总秒数和按真实总量计算的调用均值可合计，不平均分位数。

自然单case可直接归因的 span 映射真实 cohort_id、动作前stage与 tick_min/max；共享 span 保留 unknown/shared。E_grades 四case复用 state_0000–0031，而 collector key 未包含 test，故其分类计时可能混合四个距离层，case/距离/成功失败归因必须 unknown。独立 episode CSV 仍保存每case真实H5距离与成功终点，不能把这些终点附会到混合计时。

## 父进程与子进程

|任务|角色|PID/PPID|collector wall s|process CPU s|退出|
|---|---|---|---|---|---|
|E|evaluation|959513/959424|3063.267693|3037.138562|0|
|E|cpu_geometry_child|981257/959513|2976.515080|21.707138|0|
|E|cpu_geometry_child|981779/959513|2976.348588|23.335218|0|
|E|cpu_geometry_child|982291/959513|2976.132068|17.404408|0|
|E|cpu_geometry_child|982787/959513|2975.873407|17.090812|0|
|A|evaluation|2111766/959424|183.206934|76.387007|0|
|B|evaluation|2158540/959424|2931.935439|2928.216597|0|
|C|evaluation|2158890/959424|2929.999373|2935.494413|0|
|D|evaluation|2759676/959424|3085.658323|3096.984580|0|
|E_grades|evaluation|2760066/959424|10964.865768|11144.253597|0|
|E_grades|cpu_geometry_child|2779778/2760066|10868.190371|54.892921|0|
|E_grades|cpu_geometry_child|2780266/2760066|10868.629038|49.959871|0|
|E_grades|cpu_geometry_child|2780767/2760066|10868.963718|50.427156|0|
|E_grades|cpu_geometry_child|2781248/2760066|10869.246840|48.810605|0|

collector 的时间窗从独立接入开始，包含模型加载、校准、benchmark及close；不是纯episode wall。Isaac app首次启动可能在窗外。子进程CPU累计单列，不能相加成父线程wall或完整机器人延迟。

实际环境检查：父评测进程 OPENBLAS_NUM_THREADS=1、OMP=2、MKL=2；原正式v2与本诊断CPU子进程实际 OPENBLAS=32，OMP=2、MKL=2。此处记录观测配置，不将所有进程误称单BLAS线程，也不修改本轮配置。

## 同线程份额及分母

|任务/角色|PID/TID|观测exclusive wall s|几何wall占观测同线程|观测exclusive thread CPU s|几何CPU占观测同线程|几何thread CPU / process CPU|
|---|---|---|---|---|---|---|
|E/evaluation|959513/138758855829312|2739.418188|2.443%|2616.057860|2.570%|2.214%|
|E/cpu_geometry_child|981257/133938876700480|20.645587|6.786%|20.639007|6.797%|6.463%|
|E/cpu_geometry_child|981779/135854606100288|22.273854|5.020%|22.263049|5.032%|4.801%|
|E/cpu_geometry_child|982291/132893130368832|16.397466|6.153%|16.389805|6.166%|5.806%|
|E/cpu_geometry_child|982787/124075584268096|16.096002|5.471%|16.088080|5.482%|5.160%|
|A/evaluation|2111766/133090788378432|137.309066|2.301%|48.808170|6.478%|4.139%|
|B/evaluation|2158540/137261947852608|2623.361979|2.647%|2553.182403|2.735%|2.385%|
|C/evaluation|2158890/123467920791360|2622.155587|2.592%|2559.355369|2.672%|2.329%|
|D/evaluation|2759676/127009501652800|2746.383851|2.656%|2651.349366|2.765%|2.367%|
|E_grades/evaluation|2760066/130893714069312|8268.033394|2.317%|8067.149691|2.389%|1.729%|
|E_grades/cpu_geometry_child|2779778/125620505794368|52.962690|10.297%|52.948748|10.315%|9.949%|
|E_grades/cpu_geometry_child|2780266/125626292852544|48.131907|9.869%|48.118431|9.887%|9.523%|
|E_grades/cpu_geometry_child|2780767/139856084121408|48.739834|8.968%|48.722192|8.985%|8.682%|
|E_grades/cpu_geometry_child|2781248/129662567900992|47.140941|8.674%|47.125543|8.689%|8.389%|

几何标签只覆盖已注入函数：未注入的 R0/legacy 调用及 other_control 中几何仍 unknown，因此不是全部CPU几何的精确份额。最后一列使用更宽 process CPU 分母，和同线程份额含义不同。多线程行不合成一个可加wall比例。

![独立诊断观测同线程几何占比](../phase15_2/results/figures/cpu_geometry_observed_share.png)

## 类别耗时与观测排序

以下每一行属于单task、单PID、单thread；总数和 exclusive 时间为原记录总量。按每个父线程 exclusive CPU 排序列前五项，作为后续工程排查的实测依据。完整类别和分层P50/P95/P99见CSV。

|任务|PID/TID|序|类别|次数|inclusive wall s|exclusive wall s|exclusive CPU s|池化分位数|
|---|---|---|---|---|---|---|---|---|
|E|959513/138758855829312|1|other_control|517829|7886.705288|2542.544753|2540.250656|unknown: raw calls not saved|
|E|959513/138758855829312|2|tracking_check|104186|2093.761801|26.255800|26.069732|unknown: raw calls not saved|
|E|959513/138758855829312|3|safe_retreat_planning|8400|5149.606466|23.107556|22.967375|unknown: raw calls not saved|
|E|959513/138758855829312|4|joint_limit_check|498118|14.989381|14.989381|15.620418|unknown: raw calls not saved|
|E|959513/138758855829312|5|isaac_step|1200|14.680123|10.466501|6.554982|one original stratum|
|A|2111766/133090788378432|1|other_control|8223|159.969222|32.399412|32.370928|unknown: raw calls not saved|
|A|2111766/133090788378432|2|isaac_step|1200|23.791558|18.323778|13.036478|one original stratum|
|A|2111766/133090788378432|3|collision_check|9489|2.172202|1.769385|1.758545|unknown: raw calls not saved|
|A|2111766/133090788378432|4|tracking_check|3430|18.150402|0.650864|0.646083|one original stratum|
|A|2111766/133090788378432|5|joint_limit_check|13171|0.513306|0.513306|0.529526|unknown: raw calls not saved|
|B|2158540/137261947852608|1|other_control|514379|7607.132716|2476.043525|2473.339045|unknown: raw calls not saved|
|B|2158540/137261947852608|2|tracking_check|105311|2076.811151|27.550526|27.357588|unknown: raw calls not saved|
|B|2158540/137261947852608|3|safe_retreat_planning|5290|2460.996164|21.943146|21.820302|unknown: raw calls not saved|
|B|2158540/137261947852608|4|joint_limit_check|501585|15.333842|15.333842|16.025239|unknown: raw calls not saved|
|B|2158540/137261947852608|5|isaac_step|1200|17.322987|13.320600|9.624926|one original stratum|
|C|2158890/123467920791360|1|other_control|514688|7612.312798|2487.277093|2485.519545|unknown: raw calls not saved|
|C|2158890/123467920791360|2|tracking_check|105432|2072.072361|27.023219|26.840872|unknown: raw calls not saved|
|C|2158890/123467920791360|3|safe_retreat_planning|5290|2453.969273|21.149890|21.043655|unknown: raw calls not saved|
|C|2158890/123467920791360|4|joint_limit_check|501907|15.125023|15.125023|15.814631|unknown: raw calls not saved|
|C|2158890/123467920791360|5|isaac_step|1200|12.719781|8.702313|4.978925|one original stratum|
|D|2759676/127009501652800|1|other_control|514820|7810.440293|2545.099574|2543.584122|unknown: raw calls not saved|
|D|2759676/127009501652800|2|isaac_step|1200|48.364729|40.987824|33.923257|one original stratum|
|D|2759676/127009501652800|3|tracking_check|105574|2074.401639|27.821662|27.621877|unknown: raw calls not saved|
|D|2759676/127009501652800|4|safe_retreat_planning|5290|2466.245991|22.895534|22.781207|unknown: raw calls not saved|
|D|2759676/127009501652800|5|joint_limit_check|502629|16.042306|16.042306|16.722562|unknown: raw calls not saved|
|E_grades|2760066/130893714069312|1|other_control|1573071|23807.180345|7756.028723|7749.250791|unknown: raw calls not saved|
|E_grades|2760066/130893714069312|2|isaac_step|4800|169.875713|143.602051|118.304642|one original stratum|
|E_grades|2760066/130893714069312|3|tracking_check|253089|6347.380126|78.490958|77.942939|unknown: raw calls not saved|
|E_grades|2760066/130893714069312|4|safe_retreat_planning|21015|15610.501574|54.349608|53.876161|unknown: raw calls not saved|
|E_grades|2760066/130893714069312|5|joint_limit_check|1517496|45.526471|45.526471|47.592486|unknown: raw calls not saved|

待检验设计：对各线程上榜的 collision/FK/IK/joint/retreat 模块先核查重复几何构建及不变量计算，再考虑静态缓存、FK/collision批处理；tracking/Normal候选检查可设计持久池/绑核与按原clone序提交。若 top 项为 other_control，需要增加独立span归因再作模块优先级判断，不凭标签猜测。所有设计只用于未来新协议。

## 调度、子池、RPC与开销限制

v2撤离完整保留主进程串行执行；仅 tracking/Normal 前三个真实候选checks并行。dispatch span包含IPC及等待；pure IPC未独立测得则unknown。pool startup、serial shadow及operation_timing从各真实pool原件独立引用，不把整pool跨case总量复制为每case耗时。

π等待为 inference_rpc_wait，含通信/服务等待，不等于 GPU idle 或纯forward；CUDA-event未观测为unknown。Isaac step 与日志I/O独立span记录，未注入区域unknown。10角色batch1的原样本可跨样本重算分位数，与未存逐call的CPU collector不同。

计时开销只在同一实际输入20次原生 escape-edge fixture对照观察约2.57%。完整闭环没有等价未注入重复对照，闭环因果开销unknown；诊断与正式既有耗时只能描述不同scope/版本，不能推断v2或全流程加速倍率。v1的3.396倍fixture不外推。

## 科学解释与未来工作

低成功率、长CPU规划及600tick预算耗尽都保留为负证据，不是允许改本轮协议的技术故障。有限IK未解、多次规划失败不能证明数学不可达，model-clear不保证完整物理安全，实测arm/wrist collision仍unknown。Fail-fast与更长horizon只作为未来独立验证假设。

正式E随机83.125%、人工63%、自然2.8%、Full Loop82.34375%，未通过主要门槛；fixed98.4375%及CI下限95.6938%通过。CPU报告不改变无Anchor/无Phase16/无在线RL/LWD/DIVL资格的已审计结论。独立Re-acquire只建议未来公平比较GT Planner、轻量state BC与pi0.5，终点是Normal可接管状态；不自动采专家训练数据或启动训练。

本报告由关闭后的真实collector生成；报告刷新后重跑原独立auditor。审计与本地交付状态以最终delivery marker及逐项SHA为准。

## 已关闭pool的完整操作作用域

以下分位数为各自pool在原operation_timing中直接记录的完整操作分位数，未平均其他分层的quantile；两个pool不合并。E_grades整pool跨四个grade，不复制为每grade或每episode。retreat为主进程串行完整调用wall，含内部检查；不能与内层FK/collision/exclusive份额相加。比值分母是更宽collector wall（含加载/校准/close），不是纯episode时间或纯CPU比例。

|任务|操作|count|wall总秒|P50秒|P95秒|P99秒|wall/collector wall|
|---|---|---|---|---|---|---|---|
|E|retreat|2800|2574.083732|0.806016|1.197892|1.215835|84.031%|
|E|tracking|475|30.479906|0.055910|0.104543|0.108818|0.995%|
|E|candidates|144|0.350724|0.002158|0.004886|0.005883|0.011%|
|E_grades|retreat|7005|7803.258549|1.104841|1.212735|1.257916|71.166%|
|E_grades|tracking|1338|72.039353|0.038320|0.151049|0.180544|0.657%|
|E_grades|candidates|392|1.030130|0.002288|0.005612|0.006213|0.009%|

两池dispatch总wall分别30.480046秒（E自然）与72.039894秒（E四grade），串行shadow分别0.412473/0.766863秒，作用域来自原pool。纯IPC与纯CPU拆分unknown；geometry_startup按独立collector原分层记录，不能归入每episode。

## 实测热点解释与工程顺序

父线程other_control占观测exclusive CPU大头，但该标签合并Geometry.profile/world/goal、policy forward、geometry run等函数；标签中也包含几何，不能把已命名几何约2–6%的下界称为全部几何比例。sys.setprofile对高频Python/native call记录计时，span内包含回调/分配成本；没有未注入的匹配闭环对照，纯算法CPU分解与闭环开销unknown。

完整串行retreat调用在E自然占collector wall约84%，在四grade混合占约71%；它确实是该注入诊断的主要wall范围。下一轮优先对这一范围的profile/world/goal与重复候选构建做独立归因及缓存等价设计，再考虑tracking/joint热点、FK/collision批处理。当前不能断言collision单项占绝对主导，也不能从诊断与正式跨scope耗时推断加速倍率。

源H5/距离/成功终点有直接数据，CPU分类E_grades缺test键，无法分离各grade及成功/失败计时。单自然case的cohort/stage/tick分层保留；无直接cohort的span保持shared/unknown。

## 原每case共享控制/Router/Planner wall

|任务|case|clock|P50秒|P95秒|P99秒|
|---|---|---|---|---|---|
|E|natural_severe|control_latency|3.300163|13.234398|14.645459|
|E|natural_severe|router_latency|0.051709|0.092107|0.163932|
|E|natural_severe|planner_latency|5.026175|13.882673|14.650081|
|E|natural_severe|entry_check_latency|0.058782|1.042805|1.308063|
|E|natural_severe|transition_latency|0.004355|0.109206|0.111133|
|A|natural_severe|control_latency|0.033830|0.351397|0.742672|
|A|natural_severe|router_latency|unknown|unknown|unknown|
|A|natural_severe|planner_latency|unknown|unknown|unknown|
|A|natural_severe|entry_check_latency|unknown|unknown|unknown|
|A|natural_severe|transition_latency|unknown|unknown|unknown|
|B|natural_severe|control_latency|2.446993|13.417440|14.879842|
|B|natural_severe|router_latency|0.000232|0.000259|0.000285|
|B|natural_severe|planner_latency|5.128199|14.088133|14.921914|
|B|natural_severe|entry_check_latency|unknown|unknown|unknown|
|B|natural_severe|transition_latency|unknown|unknown|unknown|
|C|natural_severe|control_latency|2.484989|13.366858|14.843419|
|C|natural_severe|router_latency|0.050580|0.063044|0.064874|
|C|natural_severe|planner_latency|5.098582|14.034298|14.875928|
|C|natural_severe|entry_check_latency|0.058863|0.972151|1.215568|
|C|natural_severe|transition_latency|unknown|unknown|unknown|
|D|natural_severe|control_latency|2.846073|13.396585|14.867593|
|D|natural_severe|router_latency|0.052006|0.076635|0.232008|
|D|natural_severe|planner_latency|5.305601|14.024324|14.854896|
|D|natural_severe|entry_check_latency|0.166300|0.959807|1.577191|
|D|natural_severe|transition_latency|0.012473|0.335057|0.398945|
|E_grades|grade_05|control_latency|2.447148|12.650842|15.287850|
|E_grades|grade_05|router_latency|0.000566|0.107772|0.193022|
|E_grades|grade_05|planner_latency|3.518933|13.198983|15.372878|
|E_grades|grade_05|entry_check_latency|0.134132|0.146424|1.795653|
|E_grades|grade_05|transition_latency|0.011564|0.256188|0.590789|
|E_grades|grade_10|control_latency|0.004635|16.646476|19.847252|
|E_grades|grade_10|router_latency|0.000551|0.061922|0.164299|
|E_grades|grade_10|planner_latency|7.484717|18.611845|19.895395|
|E_grades|grade_10|entry_check_latency|0.138661|0.140286|0.140337|
|E_grades|grade_10|transition_latency|0.011602|0.254858|0.257798|
|E_grades|grade_15|control_latency|0.004645|20.750712|24.065558|
|E_grades|grade_15|router_latency|0.000525|0.001347|0.025731|
|E_grades|grade_15|planner_latency|4.586727|23.823359|24.950992|
|E_grades|grade_15|entry_check_latency|unknown|unknown|unknown|
|E_grades|grade_15|transition_latency|unknown|unknown|unknown|
|E_grades|grade_20|control_latency|0.039241|17.805639|22.100204|
|E_grades|grade_20|router_latency|0.000492|0.000586|0.000636|
|E_grades|grade_20|planner_latency|1.096669|20.811917|22.191193|
|E_grades|grade_20|entry_check_latency|unknown|unknown|unknown|
|E_grades|grade_20|transition_latency|unknown|unknown|unknown|

这些是原case JSON实际保存的wall分位数；count未保存为unknown，不假造count，不除32，不当作纯CPU计时。grade15/20模型请求为0，π推理空样本分位数unknown。CPU分层P50/P95/P99/count/mean在cpu_context_timing.csv；10独占模型/服务/RPC在exclusive_latency_all_clocks.csv。

## 原生门槛、版本与退出证据

原生4函数exact/32 child gate只验证接入；synthetic只验证工具。真实E自然及E四grade前12批tracking与Normal candidate均exact passed，两pool已关闭，8个CPU子进程final collector均exit0，12个物理/推理进程均exit0；监督959424实际/proc不存在，未发送中断信号。monitor_diagnostics SHA固定核验的最终快照passed、活跃主/CPU均0。integration_v1 EULA与长socket失败已归档；integration_v2只修复原runtime_env与独立短socket，4计时/回放源码字节同v1。

CPU source、输入、288终点、collector、pool、报告转换与图SHA见 [最终CPU证据审计](../phase15_2/results/cpu_profile/cpu_closed_audit.json)。正式252串行/48授权v2的300项耗时见 [compute_version_timing](../phase15_2/results/metrics/compute_version_timing.csv)；8正式pool作用域另列，不与诊断两pool混合。

[CPU分层P50/P95/P99/count/mean](../phase15_2/results/cpu_profile/cpu_context_timing.csv) · [类别总量](../phase15_2/results/cpu_profile/cpu_category_totals.csv) · [同线程份额](../phase15_2/results/cpu_profile/cpu_same_thread_shares.csv) · [父/子clock](../phase15_2/results/cpu_profile/cpu_process_clocks.csv) · [288独立终点](../phase15_2/results/cpu_profile/cpu_profile_episodes.csv) · [pool操作](../phase15_2/results/cpu_profile/cpu_pool_operation_scope.csv) · [每case共享wall](../phase15_2/results/cpu_profile/diagnostic_case_shared_timing.csv)。
