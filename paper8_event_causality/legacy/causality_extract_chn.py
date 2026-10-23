# 原附件中英文正则参考；正式实验请运行 ../ECE_MAIN/infer_tech_regex.py。
# 此文件保留原设计和文件依赖，不作为报告结果生成入口。


import re, jieba
import jieba.posseg as pseg
from pyltp import SentenceSplitter
class CausalityExractor():
    def __init__(self):
        pass

    '''1由果溯因配套式'''
    def ruler1(self, sentence):
        '''
        conm2:〈[之]所以,因为〉、〈[之]所以,由于〉、 <[之]所以,缘于〉
        conm2_model:<Conj>{Effect},<Conj>{Cause}
        '''
        datas = list()
        word_pairs =[['之?所以', '因为'], ['之?所以', '由于'], ['之?所以', '缘于']]
        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][3]
                data['effect'] = result[0][1]
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
    '''2由因到果配套式'''
    def ruler2(self, sentence):
        '''
        conm1:〈因为,从而〉、〈因为,为此〉、〈既[然],所以〉、〈因为,为此〉、〈由于,为此〉、〈只有|除非,才〉、〈由于,以至[于]>、〈既[然],却>、
        〈如果,那么|则〉、<由于,从而〉、<既[然],就〉、〈既[然],因此〉、〈如果,就〉、〈只要,就〉〈因为,所以〉、 <由于,于是〉、〈因为,因此〉、
         <由于,故〉、 〈因为,以致[于]〉、〈因为,因而〉、〈由于,因此〉、<因为,于是〉、〈由于,致使〉、〈因为,致使〉、〈由于,以致[于] >
         〈因为,故〉、〈因[为],以至[于]>,〈由于,所以〉、〈因为,故而〉、〈由于,因而〉
        conm1_model:<Conj>{Cause}, <Conj>{Effect}
        '''
        datas = list()
        word_pairs =[['因为', '从而'], ['因为', '为此'], ['既然?', '所以'],
                    ['因为', '为此'], ['由于', '为此'], ['除非', '才'],
                    ['只有', '才'], ['由于', '以至于?'], ['既然?', '却'],
                    ['如果', '那么'], ['如果', '则'], ['由于', '从而'],
                    ['既然?', '就'], ['既然?', '因此'], ['如果', '就'],
                    ['只要', '就'], ['因为', '所以'], ['由于', '于是'],
                    ['因为', '因此'], ['由于', '故'], ['因为', '以致于?'],
                    ['因为', '以致'], ['因为', '因而'], ['由于', '因此'],
                    ['因为', '于是'], ['由于', '致使'], ['因为', '致使'],
                    ['由于', '以致于?'], ['因为', '故'], ['因为?', '以至于?'],
                    ['由于', '所以'], ['因为', '故而'], ['由于', '因而']]

        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[p|c]+\s(.*)(%s)/[p|c]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][0] + '-' + result[0][2]
                data['cause'] = result[0][1]
                data['effect'] = result[0][3]
                datas.append(data)
        if datas:
            return datas[0]
        else:
            return {}
    '''3由因到果居中式明确'''
    def ruler3(self, sentence):
        '''
        cons2:于是、所以、故、致使、以致[于]、因此、以至[于]、从而、因而
        cons2_model:{Cause},<Conj...>{Effect}
        '''

        pattern = re.compile(r'(.*)[,，]+.*(于是|所以|故|致使|以致于?|因此|以至于?|从而|因而)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data
    '''4由因到果居中式精确'''
    def ruler4(self, sentence):
        '''
        verb1:牵动、导向、使动、导致、勾起、引入、指引、使、予以、产生、促成、造成、引导、造就、促使、酿成、
            引发、渗透、促进、引起、诱导、引来、促发、引致、诱发、推进、诱致、推动、招致、影响、致使、滋生、归于、
            作用、使得、决定、攸关、令人、引出、浸染、带来、挟带、触发、关系、渗入、诱惑、波及、诱使
        verb1_model:{Cause},<Verb|Adverb...>{Effect}
        '''
        pattern = re.compile(r'(.*)\s+(牵动|已致|导向|使动|导致|勾起|引入|指引|使|予以|产生|促成|造成|引导|造就|促使|酿成|引发|渗透|促进|引起|诱导|引来|促发|引致|诱发|推进|诱致|推动|招致|影响|致使|滋生|归于|作用|使得|决定|攸关|令人|引出|浸染|带来|挟带|触发|关系|渗入|诱惑|波及|诱使)/[d|v]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data
    '''5由因到果前端式模糊'''
    def ruler5(self, sentence):
        '''
        prep:为了、依据、为、按照、因[为]、按、依赖、照、比、凭借、由于
        prep_model:<Prep...>{Cause},{Effect}
        '''
        pattern = re.compile(r'\s?(为了|依据|按照|因为|因|按|依赖|凭借|由于)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]

        return data

    '''6由因到果居中式模糊'''
    def ruler6(self, sentence):
        '''
        adverb:以免、以便、为此、才
        adverb_model:{Cause},<Verb|Adverb...>{Effect}
        '''
        pattern = re.compile(r'(.*)(以免|以便|为此|才)\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][0]
            data['effect'] = result[0][2]
        return data

    '''7由因到果前端式精确'''
    def ruler7(self, sentence):
        '''
        cons1:既[然]、因[为]、如果、由于、只要
        cons1_model:<Conj...>{Cause},{Effect}
        '''
        pattern = re.compile(r'\s?(既然?|因|因为|如果|由于|只要)/[p|c]+\s(.*)[,，]+(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][0]
            data['cause'] = result[0][1]
            data['effect'] = result[0][2]
        return data
    '''8由果溯因居中式模糊'''
    def ruler8(self, sentence):
        '''
        3
        verb2:根源于、取决、来源于、出于、取决于、缘于、在于、出自、起源于、来自、发源于、发自、源于、根源于、立足[于]
        verb2_model:{Effect}<Prep...>{Cause}
        '''

        pattern = re.compile(r'(.*)(根源于|取决|来源于|出于|取决于|缘于|在于|出自|起源于|来自|发源于|发自|源于|根源于|立足|立足于)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]
        return data
    '''9由果溯因居端式精确'''
    def ruler9(self, sentence):
        '''
        cons3:因为、由于
        cons3_model:{Effect}<Conj...>{Cause}
        '''
        pattern = re.compile(r'(.*)是?\s(因为|由于)/[p|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]

        return data

    '''抽取主函数'''
    def extract_triples(self, sentence):
        infos = list()
      #  print(sentence)
        if self.ruler1(sentence):
            infos.append(self.ruler1(sentence))
        elif self.ruler2(sentence):
            infos.append(self.ruler2(sentence))
        elif self.ruler3(sentence):
            infos.append(self.ruler3(sentence))
        elif self.ruler4(sentence):
            infos.append(self.ruler4(sentence))
        elif self.ruler5(sentence):
            infos.append(self.ruler5(sentence))
        elif self.ruler6(sentence):
            infos.append(self.ruler6(sentence))
        elif self.ruler7(sentence):
            infos.append(self.ruler7(sentence))
        elif self.ruler8(sentence):
            infos.append(self.ruler8(sentence))
        elif self.ruler9(sentence):
            infos.append(self.ruler9(sentence))

        return infos

    '''抽取主控函数'''
    def extract_main(self, content):
        sentences = self.process_content(content)
        datas = list()
        for sentence in sentences:
            subsents = self.fined_sentence(sentence)
            subsents.append(sentence)
            for sent in subsents:
                sent = ' '.join([word.word + '/' + word.flag for word in pseg.cut(sent)])
                result = self.extract_triples(sent)
                if result:
                    for data in result:
                        if data['tag'] and data['cause'] and data['effect']:
                            datas.append(data)
        return datas

    '''文章分句处理'''
    def process_content(self, content):
        return [sentence for sentence in SentenceSplitter.split(content) if sentence]

    '''切分最小句'''
    def fined_sentence(self, sentence):
        return re.split(r'[？！，；]', sentence)


'''测试'''
def test():

    content1 = '''
爱股票华为，真正到了一个关口，生死攸关的关口。这两天，华为悲壮地宣布：由于美国制裁，9月15日之后，华为的麒麟高端芯片将无法制造，成为绝版！缺芯！少魂！这个麒麟9000芯片，采用的最先进的5纳米工艺，只有台积电能生产出来，之前与时间赛跑囤积的1300万片，卖完也就没有了。美国的禁令，在时隔半年之后，恶果终于显现，华为的高端芯片就要耗尽了，英雄末路挥泪别！今年上半年，华为的智能手机市场份额全球第一，这还是美国制裁下，少发货6000万台的情况下。如果不是美国制裁，去年就可以遥遥领先，甩出第二名好几条街！好不容易挤进了手机芯片第一梯队，如今却被美国卡了脖子，未来一段时间里极有可能再次落后。就像游泳比赛，“去年的一轮制裁相当于把我们的手捆上了，今年的第二轮制裁相当于把我们的腿捆上了。我们在扭腰扭屁股和别人比赛。”核心技术受制于人，即使心里憋着一口闷气，却又不得不接受这个残酷的现实！更严重的还在后头，如果上游芯片供应被彻底切断，华为生产便会陷入停滞！面对整个美国的重重围剿，而且是总统亲自坐镇指挥，华为这一次真的太难了。还没到9月15日，就已经开始缺货了。华为现在的困难可能比我们想象的要大！华为虽然能设计出世界一流水平的7纳米手机芯片，但国内没有一家芯片制造厂能生产出来。华为的芯片再好，离开了台积电，也只能停留在图纸上！所以现在华为手中的芯片，如同战场上的弹药一样，打一发少一发，直到用完。智能工业时代，没多少产品可以离开芯片。中国作为世界上最大的芯片进口国，每年要吃进全球超过一半的芯片。华为这样的电子科技巨头，拿钱却买不到高端芯片，就好比大工业时代，能制造汽车、轮船、飞机，却没有发动机。三年自然灾害时期，有钱没地方买粮食！芯片供应不上了，华为怎么办？01华为和苹果一样，其实自己设计芯片，但不直接生产制造芯片。只负责画图纸，不负责生产，图纸画出来后，就得找芯片代工厂生产。是负责将图纸落地的芯片代工厂，他们像建筑施工队，要做的是按照图纸地把房子盖出来。这方面，全世界最牛的企业，是台积电，规模和技术均列全球第一。咱们的中芯国际，最先进工艺是14nm工艺，大概是台积电5年前的水平，直接被干趴下！但台积电再牛逼又能怎样，用是美国的技术。只要美国一纸禁令，台积电就不能给华为供货了。美国可以利用手里的霸权，一分钱不花就凭空制造出无数手段来打压你。只要你没有芯片制造能力，就处处被动。这次是台积电，下次就可能是高通、联发科，卡死芯片供应商，你的订单就全部废了。美国现在是尽一切可能，要切断华为的所有供应链，全球任何公司在这样的环境下，都要抓瞎，只有华为能坚持下去。这是一场不得不打的战争，因为特朗普根本没想给我们活路，他的目的只有一个，搞死搞残中国的高科技产业。接下来，美国封杀影响可能会更加明显、致命。所以中国的下一步棋，就是去美国化，自研自产芯片，突破美国锁喉！02这是一场不对称的战争，也是我们最痛苦的战争。众所周知，做芯片堪比做原子弹，甚至更难。原子弹的研发，是0和1的过程，做不出来就是0，做出来就是1。做不出来啥都不是，做出来就是王者。但芯片不但要做出来，还要做到世界顶尖，千年老二没有用，连口汤都喝不上。举个，如果原子弹市场化，所有的国家都可以直接买，苏联的所有优势就没有了，大家都认为美国的原子弹更先进，苏联的原子弹就卖不出去。同理做芯片，需要保持国际上最先进的科技水平，否则，你做出来别人也不用你的。这就叫赢家通吃！在芯片代工上，台积电一直稳定保持最强大的制作工艺，所以一家独大！但是，全球没有任何一家公司能够独立做出芯片出来，毕竟需要全球最顶尖的产业供应链，就要求最强大的工业基础！相当于，中国需要在两年时间，以一国之力挑战全球最强的10个以上的国家联盟的40年的科技沉淀，这就是难度。或者说中国一个国家，挑战全球科技总和超过40年的沉淀。这就是难度。中国虽然是制造业大国，但高端制造方面，还需要补课！现在阶段，我们还处于被动防守、埋头苦干的防御性阶段，毕竟历史欠账太多。咱们的工业基础，太薄弱了！要补上这些课并不容易。别的不说，制造高端芯片必需的光刻机咱们就没有。但现在美国卡脖子，中国赶鸭子上架也要搞，不搞就是死路一条。03于是，华为的“南泥湾”计划诞生，全面去美化，规避美国技术、美国产品，用国产的供应链或者其他元件全面替代美国产品。说白了，你美国全面制裁封锁华为，华为就全面去美国化。其实，这本质上就是华为与美国政府的一场“科技冷战”。最近任正非亲自出马，三天拜访四所高校，想必在憋大招。华为已招聘了数百位国内顶尖人才，准备自己搞光刻机！然后自建像台积电这样的芯片生产厂。华为憋着一口气，为了活下去，只能艰难长征！跟红军一样，一边长征，一边还要攻城略地通过敌人补给自己。这年头没人不想拿巨额订单挣快钱，不要以为美国是铁板一块，内部也有利益纷争。这不，听说华为采购了美国联发科1.2亿颗芯片，旁边的高通也眼红了想加入！据美媒消息，高通正在游说美国政府允许其向华为出售芯片。资本方正在向特朗普施压，华为禁令“不仅不会阻止华为获得必要的零部件，反而可能会将数十亿美元的市场拱手让给高通的海外竞争对手”利用好敌人的内部矛盾，就能给自己制造喘息机会！争取时间就是争取生存空间。华为要将芯片设计、生产、包装、测试全部捏在自己手里，全部独立完成，不依靠上下游代工企业。根深才能叶茂！今天美国人不愿意卖给中国芯片，明天美国人就是想卖，也未必有人要了！中国政府也已经做了巨大投入，芯片企业直接10年免税，国家的资金和民间资本疯狂涌入芯片领域。如此大干快上干上几年，芯片制造的工艺突破不了？肯定可以！一旦华为自己有了制造芯片的能力，华为芯片将彻底打开产能，那高通就真的多了一个强大的对手，到那时真的要哭的还是美国，因为制造业在中国！砸钱，砸科学家，砸人才…也许这是科技史上最悲壮的长征，也是14亿人艰苦卓绝的战争！但终有一天，我们必定弯道超车，绝地反击。
高看天下事美国的制裁大棒让华为举步维艰，作为“回报”，为何中国不能对等实施反制，禁止苹果和特斯拉公司在国内开拓市场呢？对此，外国网友给出了答案。2019年5月，特朗普将矛头对准中国企业华为，并着手对其实施全面制裁。往后两年的时间里，华为对外业务开拓可以用举步维艰来形容，美国政府不但禁止国内高科技企业继续为华为提供技术支持，更是在全球范围内扼杀华为市场；拆除华为5G基站，禁止运营商使用华为产品。在近乎疯狂的制裁下，华为迎来了最黑暗的发展时期。在2019年之前，华为靠着卓越的技术能力，在国内手机市场占据了半壁江山，在欧洲市场也快速进军。如果没有“断芯”这事儿，华为产品的海外扩展率都已经超过了苹果和三星，稳居全球第一；因为华为手中有大量的5G专利，海外公司想要使用5G服务，还得向华为下订单。可是在美国制裁后，华为5G海外业务变得停滞不前。巅峰时期华为海思全球市场份额超过30%，但是在经历了美国三轮制裁大棒后，华为海思份额已经跌至0.4%，明明5G技术领先全球，华为却被迫发布4G手机营生，这本身就是一个巨大的“笑话”。可如果站在国际贸易的角度来看，美国接连对华为挥动制裁大棒，就真的合法吗？答案自然是否定的，美国对华为的制裁严重违反了世贸准则，中国完全可以在规则的基础上对美国实行同等反制。但中国却没有这样做，这其中有何缘由？许多美国网友给出了自己的看法。一些网友认为：中美之间最大的差异在于思想，当美国看到别人成长较快时，总想着用强权压制他人，以免影响到自身地位。但中国在看到他人强大时，想到的却是如何引以为鉴，来提高自身创新力与核心竞争力，争取在某一天通过自身努力赶超对方。正是思想观念的不同，中国才没有限制美国企业在中国的发展。毕竟世界那么大，市场也那么大，任何想要独享垄断专权的行为，都将受到严重惩罚，与其费尽心思制裁他人，不如想着如何一起做大。美国政府的狭隘民族观始终想不到这点，所以他们宁愿增加上百亿美元的开销，也要想方设法遏制华为的发展。但这只能享受一时的效果，从另一方面来说，美国的制裁也迫使华为走上了一条自主研发的道路。往年华为在电子产品领域多数依赖外国进口的芯片，可是自从被美国卡脖子后，华为电子产品研发业务差点被腰斩。也是从这时候华为才意识到，如果不能掌握核心技术，华为迟早有一天会失去竞争优势，所以哪怕自主研发这条路非常难走，也一定要坚持下去。美国制裁华为其实是伤敌一千自损八百，美国在5G上的额外开销高达数百亿，根据美国网友的说法，美国供货商每年对华为的出口损失达到了200亿美元。只要美国企业在中国经营合法，中国是不会轻易制裁美企的。中国一直坚持合作共赢的原则，外企进入中国市场不但能提供就业岗位，更容易在行业内形成竞争机制，促使国内企业学习外企的长处，不断改善自身实现创新与进步。中国甚至为特斯拉提供优惠贷款，当然，这份自信和气度，是美国永远学不会的。
    '''
    with open('122.txt', 'rt', encoding='utf-8') as f:
        # open为打开文件，r为读取
        f = open('122.txt', 'rt', encoding='utf-8')
        # 逐行读取文件内容
        lines = f.readlines()
    with open("test.txt", "w", encoding='utf-8') as file:
     for line in lines:
      extractor = CausalityExractor()
      datas = extractor.extract_main(line)
      for data in datas:
        print('******'*4)
        file.write('******'*4)
        file.write("\n")
        print('cause', ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]]))
        file.write('cause')
        file.write(''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]]))
        file.write("\n")
        print('tag', data['tag'])
        file.write("\n")
        file.write('tag')
        file.write(data['tag'])
        print('effect', ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]]))
        file.write('effect')
        file.write(''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]]))
        file.write("\n")

test()
