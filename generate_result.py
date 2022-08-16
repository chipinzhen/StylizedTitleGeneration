# -*- coding: utf-8 -*-
import torch
from model import generateModel
import torch.nn.functional as F
import random
import jieba
import json


def loadModel(checkpointPath):
	net = generateModel()
	checkpoint = torch.load(checkpointPath)
	net.load_state_dict(checkpoint['model_state_dict'])
	return net

net = loadModel(checkpointPath='./model_lambda2_epoch1.pt')
net.eval()
net.to('cuda:0')
# print(net)



from transformers import AutoTokenizer
from transformers.generation_utils import top_k_top_p_filtering

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

def generateHeadline(model, input_news, task_id, tokenizer, top_p):
	data = tokenizer(input_news, padding=True, truncation=True)
	input_ids = torch.Tensor(data['input_ids']).view(1,-1).long().to('cuda:0')
	attention_mask = torch.Tensor(data['attention_mask']).view(1, -1).long().to('cuda:0')
	label = ['']
	tokens = []
	i = 1
	with torch.no_grad():
		while True:
			result =  tokenizer(label, padding=True, truncation=True)
			output =  model(input_ids=input_ids, attention_mask=attention_mask,
							labels=torch.Tensor(result['input_ids']).to('cuda:0').view(1, -1).long(), task_id=task_id)
			logits = output[1][0]
			logits = top_k_top_p_filtering(logits=logits, top_p=top_p)
			probabilities = F.softmax(logits[i], dim=-1)

			i2 = 0
			while True:
				next_token = torch.multinomial(probabilities, 1)
				i2 += 1
				if (next_token.to('cpu').numpy()[0] != 100) or i2 >= 10:
					break

			if next_token.to('cpu').numpy()[0] == 102:
				break

			label[0] = label[0] + tokenizer.convert_ids_to_tokens([next_token.to('cpu').numpy()[0]])[0]
			tokens.append(tokenizer.convert_ids_to_tokens([next_token.to('cpu').numpy()[0]])[0])
			i += 1

			if i >= 128:
				break

	sentence = ''.join(list(filter(filterUNK, tokens)))
	return sentence



def filterUNK(token):
	return token != '[UNK]'

# input_news = '记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的训练，全队立即返回酒店。在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出现被队伍放在了相当重要的位置。而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。从长春到沈阳，雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种事情。”一位国奥球员也对雨水的“青睐”有些不解。'
# input_news2 = '体坛周报特约记者张锐北京报道 谢亚龙已经被公安部门正式宣布“立案侦查”，但谢亚龙等人是哪一天被辽宁警方带走的？又是如何被带走的？外界有着很多种版本。但来自内部的消息显示，谢亚龙等人被带走，其实是与南、杨被带走时如出一辙。9月3日(周五)下午4点左右，谢亚龙在中体产业办公室里接到了一个电话。这个电话来自于国家体育总局，要求他赶往总局参加一个会议，至于会议内容并没有说明。对下属打了一声招呼之后，谢亚龙便出了门。到了总局之后，谢才了解到“会议的内容”——辽宁警方在等待着他。没有任何预兆，谢亚龙被警方带到了沈阳。这样的方式，与年初南杨二人被带走如出一辙。当时，南杨也是接到总局领导的电话，要求连夜去参加一个紧急会议，一到总局便被辽宁警方带走。换而言之，警方带走谢亚龙等人，总局领导事先是知情的。而中体产业则并不知情，他们起初接到的信息谢亚龙仅仅只是去总局开会。所以，中体产业没有在第一时间发出谢亚龙协助调查的通告，后来在证监会的要求下才被动发出。外界一直猜测谢亚龙是协查还是自己有事，直到12日上午才有了答案，公安部网站“治安新闻”中出现《警方证实谢亚龙等被立案侦查》一文：“在侦办涉嫌利用足球比赛贿赂、赌博系列案件中，公安、检察机关多渠道获取对原中国足协副主席谢亚龙、原中国足协裁判委员会主任李冬生、原国家足球队领队蔚少辉等人涉案的线索和举报。在国家体育总局的配合下，专案组已依法对谢亚龙、李冬生、蔚少辉立案侦查。”当天中午央视新闻频道的新闻节目中，也报道了谢亚龙等人被立案侦查的消息。下午5点左右，各门户网站刊发了中国足协的声明：“公安机关依法对谢亚龙、李冬生、蔚少辉立案侦查，这是彻底整肃足坛、依法打击足球假、赌、黑治理行动的继续。足管中心坚决贯彻落实总局‘治理足球假、赌、黑违法犯罪行为，坚决惩治腐败，整顿足坛，坚决支持配合公安部门依法打击足球假、赌、黑违法犯罪行为，绝不姑息’的要求，做好自身工作。”距谢亚龙等人被警方带走，已经有10天。根据相关的法律，第14天(也就是9月16日)将会是一个“坎”，因为如果问题严重，在证据确凿的情况下，“立案侦查”将有可能演变为“正式逮捕”。'
# input_news3 = '本报讯(记者/戴远程实习生/陈颖)一个是享有“中国牛仔看新塘”之美誉的全国著名牛仔专业市场，一个是致力于“让天下没有难做的生意”之全球领先电子商务服务提供商，两个看似互不相关的行业强势联手，传统专业市场正利用电子商务企业的技术创新优势为“广货北上”的创造新途径。昨天下午，新塘牛仔企业战略联盟新闻发布会在广州举行，阿里巴巴(中国)网络技术有限公司与增城市新塘镇人民政府、新塘国际牛仔城、新塘牛仔文化传播公司缔结战略合作协议，为新塘牛仔企业搭建线上产业集群电子商务平台，助其寻找商机、拓展经营范围、优化商业模式。此次签约也标志着阿里巴巴与广州专业市场战略合作的全面突破。“借助电子商务平台进行业务拓展被列为新塘牛仔的重要发展战略。”新塘国际牛仔城、新塘牛仔文化传播有限公司董事长陈康强昨日向记者表示，牵手阿里巴巴还也有利于提升新塘牛仔的专业市场品牌，积极通过“广货北上”拓展国际国际新兴市场。'
# input_news4 = '晨报记者 苗夏丽“按CRIC上市首日的市值计算，新浪的资产增加6亿美金。 ”新浪CEO曹国伟日前在接受媒体采访时表示。继搜狐畅游、盛大游戏等分拆登陆纳斯达克之后，新浪与易居中国的合资公司CRIC(中国房产信息集团)美国时间16日也顺利实现在纳斯达克的分拆上市，这是中国首家在纳斯达克上市的中国地产科技概念股，也创造了两家在美国上市的中国公司分拆各自业务后合并，并进行二次上市的先河。新浪资产增加6亿美金CRIC去年2月由易居旗下的房地产信息咨询业务与新浪房地产网络业务合并而成，是中国最大的专业房地产信息服务公司，拥有同时覆盖线上线下的房地产综合信息和服务平台。据了解，此次CRIC首次公开发行1800万股美国存托凭证，定价为每股12美元，募资2.16亿美元；同时承销商有权在未来30天内行使总额达到270万股的超额配售权。发行后，控股股东易居中国持股51%，新浪作为第二大股东持股33%。上市第一天，CRIC以12.28美元开盘，尾盘报收14.2美元，逆市上涨18.33%，这在曹国伟看来是“又一个成功故事的开始”。“CRIC首日市值近20亿元，与新浪市值相差无几。”曹国伟接受记者采访时称，新浪所有的房地产业务将全部由新上市公司CRIC来主导，但这不会影响新浪的收入，一方面按新浪的持股比例和CRIC的市值计算，新浪资产增加6亿美金，同时CRIC未来实现的净利润也有三分之一并入新浪。仅仅是个开始对新浪和曹国伟来说，CRIC的成功上市应该只是一个开始。“新浪会继续维持门户、内容和广告等核心竞争力，同时尝试多元化利润来源。”曹国伟表示，新浪容易被忽略的是，它不仅是一个门户网站，实际上在垂直领域拥有众多优势，蕴含商机，“如果有合适的时机、合适的合作伙伴以及合适的商业模式，新浪会考虑其他业务的未来发展。”实际上，除了房产业务，汽车、股票、游戏、数码等都是新浪市场和收入很大的垂直内容领域，都有可能诞生新的上市公司，到时新浪可能更像是一个类似孵化器的公司。当然可能存在其他门户网站的同质化竞争、合作伙伴的选择和磨合等问题，但是CRIC的上市已经成为新浪的一个范本，似乎也为新浪的未来发展战略指明了一个方向。'
# print(generateHeadline(model=net, input_news=input_news, task_id=0, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news, task_id=1, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news, task_id=2, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news, task_id=3, tokenizer=tokenizer, top_p=0.3))
# print('---' * 20)
# print(generateHeadline(model=net, input_news=input_news2, task_id=0, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news2, task_id=1, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news2, task_id=2, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news2, task_id=3, tokenizer=tokenizer, top_p=0.3))
# print('---' * 20) 
# print(generateHeadline(model=net, input_news=input_news3, task_id=0, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news3, task_id=1, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news3, task_id=2, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news3, task_id=3, tokenizer=tokenizer, top_p=0.3))
# print('---' * 20) 
# print(generateHeadline(model=net, input_news=input_news4, task_id=0, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news4, task_id=1, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news4, task_id=2, tokenizer=tokenizer, top_p=0.3))
# print(generateHeadline(model=net, input_news=input_news4, task_id=3, tokenizer=tokenizer, top_p=0.3))

# # def generateSummary(news, task_id):
# inputs_list = ['原标题：北欧户外生活流行趋势￨Fenix携旗下4大品牌亮相ISPOSHANGHAIFenix携旗下4大品牌亮相ISPOSHANGHAI，带来北欧户外生活流行趋势~7月5日，ISPOSHANGHAI2018亚洲（夏季）运动用品与时尚展在上海新国际博览中心拉开帷幕。展会期间瑞典飞耐时集团全球销售副总裁AlexKoska先生接受了众多媒体的采访，不仅向大家透露新品的故事，同时更为中国大陆户外爱好者们带来一个好消息——2019FjallavenClassic品牌经典的徒步活动即将登陆中国大陆，请大家敬请期待！瑞典飞耐时集团旗下4大品牌品牌：瑞典皇室御用品牌Fjällräven,德国顶级户外鞋品牌HANWAG，瑞典顶级炉具灯具品牌Primus，美国军方供应商Brunton都将在展会隆重亮相。吸引了众多户外爱好者与业内人士驻足。Fjällräven是来自瑞典的高端户外品牌，自1960年创建至今50多年的时间，Fjällräven始终致力于为热爱户外运动的人们提供功能强大、性能可靠、经久耐用的户外服装和装备，激发人们融入大自然，实现人和自然的和谐相处，尽享户外运动的乐趣。Greenland系列全线升级，除了调整版型，也改良了更多细节，并推出新的颜色选择。防水且超强耐磨的新型环保材料Bergshell是Fjällräven采用防破裂平滑工艺由尼龙材料加工制作而成的新型材料。2019年春夏季开始，Bergtagen、Keb和Ulvö背包将全部采用此种材料，以满足使用者们的不同需求，为他们尽情享受大自然提供有力的装备保障。Kanken必须是整个运动展里的焦点，各种size让人应接不暇，2019我们kankenART系列更是推出了与艺术家的合作款，想要了解艺术家合作款，请持续关注最新动态~Hanwag用专业性打动了很多来客，轻量化户外运动鞋是你远足的最佳伴侣。Primus炉具专为野炊而生，primusKuchomaGrill因其有着紧凑烤架，可单手轻松带走，并且能在户外进行BBQ等功能性，而获得了ispo大奖。Ispo与天猫联合的走秀活动中，Fjallraven将运动元素与时尚融合，刮起一股户外摩登风潮，用高颜值证明户外也可以有型有款。',
# "自从十年前乔布斯发布第一台iPhone开始，苹果手机就已经统治世界，成为了手机市场的标尺。那么，面对强大的苹果公司，国产手机的oppo、小米、华为都是怎么做的呢？根据数据显示，现在很多年轻大学生在购买手机的时候不是考虑苹果就是考虑买oppo，可见oppo的营销是非常成功的，即通过年轻人常用的拍照功能、在乎的手机外观创新满足年轻人的爱好，再不停的在年轻人喜欢的电视和明星打广告。不过，这样的副作用就是oppo真实的手机配置并不强大，还落得了高价低配、厂妹专用的名声。小米的做法不同于oppo，由于苹果是全世界最赚钱的手机，小米就打造自己成为“全世界最不赚钱的手机”，用极高的性价比吸引大量对价格敏感的用户前来购买。不过，由于小米手机低端切入，所以高端产品一直不太成功。华为的做法则是有点类似三星，在高中低端分布不同的产品线，并且设立互联网专供的荣耀系列与小米展开竞争，同时高端的mate系列则不断推出研发创新，尤其现在的mate10采用自主研发的麒麟970芯片，其具有的AI智能已经超越苹果了！那么，你觉得三家手机谁最有可能超越苹果，引领国产手机新潮流呢？",
# "很多新手淘客或者是大咖淘客都觉得做返利机器人，只要每天兢兢业业引流，不要弄虚作假，客源就会越来越多，然后进行裂变，坐拥几十万几百万粉丝，躺着赚钱，被新人尊称为大神。这都是错的，不值一提。淘宝客的本质就是信息差，在法律或者道德约束下进行的有限欺骗。99%网赚都是忽悠，所谓的增加用户粘性和提高维度都是笑话，花费那么多精力和金钱，只为了提升0.1的体验，得不偿失。为什么商家要做优惠券？一件商品200元，用了优惠券100元。淘客帮忙派发这些优惠券，我们的粉丝会拼命买下这个商品，不懂的淘宝用户直接就200原价购买了。给懂的人卖的便宜，不懂的人卖的贵，穷人和富人的钱都赚了，利益最大化。只有傻逼才以为商家是在给我们用户优惠，粉丝买到的是亏本商品。打折营销？买两件，打八折，没有吸引了。不如提高价格，实行买一赠一的销售策略。爆款引流？淘宝商家有些商品会非常便宜，甚至佣金高达90%，通过淘客打造爆款商品，吸引更多的流量，让用户去买店内其他有利润的商品，爆款商品其实是一堆没用的商品，但是用户因为便宜就会直接买单，产生冲动消费。网赚实质就是通过各种忽悠，影响用户的思维，让他们瞬间就产生消费欲望。一个街头小伙在路上抢劫了一名少女，致人死亡，判刑20年；一名某村长，盖个章卖块地，上亿的资产到手。同样是抢劫欺诈，后者才是人类智慧的结晶。知识即美德，蠢货即作恶。有美德的人，互联网将给与奖励，用欺诈赚钱的方式奖励给大咖淘客，让这类大神走入名人堂；在作恶的人，互联网将给与惩罚，用上当受骗的形式惩罚大蠢货，让这类人陷入屌丝行列。小白的智商难以匹配巨大的财富，这些钱就会给某些包装的大神以培训、投资等形式收割，优胜劣汰，互联网的机制就会越来越完善。",
# "在网贷监控这个问题上，一直存在着许多问题。近日，在短短10天之内，网上就传出了30家公司网贷踩雷的消息，涉及面已经遍布上海杭州深圳。7月以来，互联网金融行业风生水起，多个网贷平台都出了问题，要么是提现苦难，要么是负责人跑路，要么是产品违约，被立案调查的都不再少数，很多平台都是百亿级别的。深圳互金平台钱爸爸出现提现困难。公安部门已经介入了调查，根据公开信息，这个平台累计成交了325亿。多多理财官方微信号发布公告称，平台情况已失去控制，平台控制人疑似跑路。上海鸿翔银票网互联网金融信息服务有限公司实际控制人易某某投案自首，称公司的集资经营活动因资金链断裂，已无法向投资人进行兑付。其他的小平台就不胜枚举了。这次行业的大乱象有两个特点，一个是数量多，过去半年，几乎每个月都有平台踩雷，但都是个别例子，而数量如此之多，则有系统性风险的嫌疑。另外一个是呈现了区域性爆发，比如杭州和深圳就成为了重灾区。这说明这些平台的资金去向可能有重叠的地方，很多担保可能也有互保的现象，金融行业的风险交织在一起，传统的风控模型无法识别。绿能宝的推广经理在年初的时候就知道平台有资金链断裂的迹象，但为了维持平台的资金周转，在没有办法的情况下，只能采取了这张“无限复投”返现方式。这几乎就成了压死绿能宝的最后一根稻草，在后面资金没有到位的情况，这种“无限复投”推广方式彻底把绿能宝推向了深渊，不少投资人也在这场盛宴中，既成为了“凶手”的帮凶，又成为了受害者。终于在4月17日爆发，全面逾期，然后就是平台逾期的套路了，公告一个个地发，但都成了空头支票。至今，绿能宝仍然每天坚持在兑付，由于没有大额资金进场，绿能宝投资人仍然被深深地套牢着。总结一句：复投的坚决不能碰。6月，自称有央企背景、号称交易量有800亿元的唐小僧雷了。资料介绍，唐小僧2015年5月5日上线，不到一年的时间成交额就破百亿。截至2017年8月，唐小僧注册用户数已达到1000万，交易额超750亿元。自成立起，唐小僧便一直被质疑声包围。多次被曝设资金池、自融、融资造假、错配等消息。更惊人的是，继唐小僧爆雷不久之后，民间四大高返平台唯一的幸存者联璧金融最终难逃一劫，也发生爆雷了。至此，四大高返平台——钱宝网、雅堂金融、唐小僧、联壁金融全部阵亡。钱满仓也是6月雷的，猛一看，这个位于北京有上市公司做背景的平台雷的毫无道理，但是，平台雷掉最重要原因就是这个看起来很牛的背景——上市公司股东。钱满仓公司的股东有星河世界集团和天马轴承集团股份有限公司，天马轴承股票代码却在早前就被标记st，面临退市风险。钱满仓出问题可以概括成一句话：老子害死儿子，上市公司股东玩死了平台。事实证明，有上市公司做背景也不一定靠谱的，在入手平台时，一定要考察清楚平台的股东是否真有实力，不能一看到上市公司就手热。抛开行业动荡，爆雷的平台自身确实是有原因的。真金白银投进去，如果我们能在入手前仔细辨别，认真考察清楚平台的真实背景和实力，不踩雷并不难，难的是我们有足够的细心和耐心，打赢这一场雷潮战。",
# "摄影技巧初接触单反时，都会看到一个矩形的方框，很多人都会说这是直方图，但是很少人会知道这个直方图怎么看，怎么通过直方图了解当前拍摄的照片，有没有欠曝或者过曝的嫌疑，直方图是不是就能说明图片的曝光是否正确，针对上面很多人的疑问，下面就来讲一下什么是直方图，如何看懂直方图。什么是直方图？所谓的直方图，其实是一个类似柱状图的图表，图表被划分成5个区域，从左到右分别是黑色、阴影、曝光、高光、白色。其实不难理解，直方图从左到右的数值就是画面从暗到亮的渐变过程。而直方图上有像一个接一个山峰状的纵轴，指的就是有多少个像素，在这个区域里面，像素越多，山峰越高。看不懂没关系，我们通过图片实例来解析给大家听。在这张图片上，我们可以大致分为树木阴影区、远处建筑的正常曝光区、天空的最亮区。有点黑黑的地方就是阴影区，看上去是比较暗的，阴影区的像素不多，证明其面积不是很大。最明显的是最右边天空的部分，占的画面比例较大，在直方图上该区域的山峰也是最高的。这样看可能大家也是一知半解，最好的学习方法是多看照片和直方图，那么问题来了，怎么看呢？直方图在哪里看？在相机上，我们可以通过按下实时取景模式按钮，然后按INFO键调出直方图，就在屏幕的右上角。或者拍完照片后，按照片回放按键，选择你想要查看的照片，然后按INFO键调出直方图，也在屏幕的右上角。还有很多图像处理软件都会支持直方图预览，一般会在右上角显示。怎么看懂直方图？终于直入正题了，下面通过几张不同曝光的照片，教大家如何看懂直方图，还有直方图跟曝光的关系：从上图看可以知道，在画面偏暗的直方图，上面的像素都堆积在左侧。如果图像的信息跟左侧边界“接壤”了，那么就说明有的地方已经呈现一片纯黑。需要将黑色区和阴影区调亮，把像素往右移动。在过曝的照片上，直方图上大量的像素都堆积在右侧，跟上面的意思相反，就是这照片大量像素都是偏亮的，如果跟右侧边界“接壤”了，则说明有地方已经纯白一片。想要拍摄完美曝光的照片，在直方图上看是左右两边的像素几乎为零，或者不“接壤”，几乎所有像素都堆积在中央区域，那就说明所有地方都是有信息的。不过在拍摄光线比较复杂和光比较大的场景时，直方图的曲线会很乱，有可能会分左右两个极端，这时候应该拍摄几张不同曝光的照片，以此来获得更大的后期修图空间，来做HDR效果的照片。全文总结由于很多人都会说在相机屏幕上看是非常好看明亮的，但是放在电脑上曝光变暗了，有很多原因导致这个问题出现。在阳光底下看照片，相机屏幕会自动调亮，看上去照片会偏亮，其实是不准确的。通过看直方图了解当前照片的曝光情况，实时调整曝光策略，才不会翻车。往期佳作·惊艳再现新闻汇：黑卡RX100M5A、尼康全幅单电！小编这么拼，你不点赞就走么",
# "VRZONE新宿店月收入近1400万日元，年客流达50万从2017年7月开始运营的VRZONE新宿店，运营至今已近一年。据悉，其截止到目前的总接待人数达50万人，人均消费5500日元（约326元人民币），最高日客流达到2000人。目前新宿店中共有19个体验项目，其中非VR项目3个。自VRZONE将高达、最终幻想、马里奥赛车、EVA、攻壳机动队等超级IP悉数用到VR中之后，这次其还将SE（SQUAREENIX）旗下的《勇者斗恶龙VR》也加入进来。VRPinea独家点评：一家如此成功的VR体验店，绝非偶然，小编猜测应该有一些与VR无关的因素在内。英国VR创业公司ImmotionGroup于伦敦上市，计划融资575万英镑日前，专注于VR体验的VR创业公司ImmotionGroup，于伦敦证劵交易所创业板AIM挂牌上市，发行价10便士，发行5750万股，计划融资575万英镑。ImmotionGroup是一家专门打造沉浸式娱乐体验的初创公司，其主要产品包括硬件式平台VRCoprs和V-Racer等。其VRExperiencePlatforms允许用户体验动态效果，以及高质量的音效和视觉图形，从而提供优于当前家庭VR的顶尖体验。VRPinea独家点评：若融资成功，将是大众购物和娱乐的又一福音。谷歌Daydream或将新增多控制器支持日前，有Reddit用户发现，谷歌Daydream和Cardboard设备的GoogleVR软件增加了多控制器支持，但具体哪款设备会提供此支持尚不清楚。如果Daydream头显可以使用多个控制器，这将使玩家与游戏之间的交互变得更加容易。VRPinea独家点评：简便的交互模式，将带给玩家更棒的游戏体验~《CocoVR》、《RickAndMorty》等7部VR作品获艾美奖提名日前，第70届艾美奖公布提名名单，其中有7项VR作品入选。《重返月球》、《银翼杀手2049:记忆实验室》、《CocoVR》、《NASAJPL：卡西尼号的最后使命》、《蜘蛛侠:英雄归来》角逐的是最佳原创交互式节目奖；《瑞克和莫蒂:虚拟科学狂人瑞克》、《硅谷：互动世界》角逐的是剧情类交互式媒介突出创意成就奖。VRPinea独家点评：大家想pick哪一部呢？《PaperDolls》上线造梦科技VR游戏平台《PaperDolls》是一款恐怖冒险密室解谜VR游戏。该款游戏精心打造了极度逼真惊悚的百年古宅，展现独特的东方文化。这样的一款适合国人的恐怖游戏，目前已经以精简版的形式上线ZMVR（造梦科技平台）。据悉，《PaperDolls》上线不到三个小时，就取得了近千份的销售量，一周之内便覆盖了五千多家VR体验店。VRPinea独家点评：战胜恐惧，一起来一场中国式冒险吧~"]

inputs_list = [json.loads(line) for line in open('thucnews.json', 'r', encoding='utf-8')]
random.shuffle(inputs_list)
inputs_list = [x['article'] for x in inputs_list][0:3000]

count1 = {}
count2 = {}
count3 = {}

output_news_and_titles = []
for new in inputs_list:
	# print('---' * 20) 
	# print(generateHeadline(model=net, input_news=new, task_id=0, tokenizer=tokenizer, top_p=0.95))
	# print(generateHeadline(model=net, input_news=new, task_id=1, tokenizer=tokenizer, top_p=0.95))
	# print(generateHeadline(model=net, input_news=new, task_id=2, tokenizer=tokenizer, top_p=0.95))
	# print(generateHeadline(model=net, input_news=new, task_id=3, tokenizer=tokenizer, top_p=0.95))
	try:
		res1 = generateHeadline(model=net, input_news=new, task_id=1, tokenizer=tokenizer, top_p=0.95)
		res2 = generateHeadline(model=net, input_news=new, task_id=2, tokenizer=tokenizer, top_p=0.95)
		res3 = generateHeadline(model=net, input_news=new, task_id=3, tokenizer=tokenizer, top_p=0.95)
	except:
		continue

	dict_tmp = {}
	dict_tmp['news'] = new
	dict_tmp['res1'] = res1
	dict_tmp['res2'] = res2
	dict_tmp['res3'] = res3

	with open('输出结果.json', 'a+', encoding='utf-8') as f:
		json_str = json.dumps(dict_tmp, ensure_ascii=False)
		f.write(json_str)
		f.write('\n')
	res1_list = jieba.cut(res1, cut_all=False)
	for w in res1_list:
		count1[w] = count1.get(w, 0) + 1

	res2_list = jieba.cut(res2, cut_all=False)
	for w in res2_list:
		count2[w] = count2.get(w, 0) + 1

	res3_list = jieba.cut(res3, cut_all=False)
	for w in res3_list:
		count3[w] = count3.get(w, 0) + 1

sorted1 = sorted(count1.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
sorted2 = sorted(count2.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
sorted3 = sorted(count3.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

with open('词频结果1.txt', 'w', encoding='utf-8') as f:
	for item in sorted1:
		f.write(str(item[0]) + ':' + str(item[1]) + '\n')

with open('词频结果2.txt', 'w', encoding='utf-8') as f:
	for item in sorted2:
		f.write(str(item[0]) + ':' + str(item[1]) + '\n')


with open('词频结果3.txt', 'w', encoding='utf-8') as f:
	for item in sorted3:
		f.write(str(item[0]) + ':' + str(item[1]) + '\n')

