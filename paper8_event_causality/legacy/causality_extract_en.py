# 原附件中英文正则参考；正式实验请运行 ../ECE_MAIN/infer_tech_regex.py。
# 此文件保留原设计和文件依赖，不作为报告结果生成入口。


import re
# import pse

import nltk
import nltk.data
from nltk import pos_tag
from nltk import RegexpParser




class CausalityExractor():
    def __init__(self):
        pass

    # '''1由果溯因配套式en'''
    # def ruler1_en(self, sentence):
    #     '''
    #     conm2:〈is,because〉、〈[之]所以,由于〉、 <[之]所以,缘于〉
    #     conm2_model:<Conj>{Effect},<Conj>{Cause}
    #     '''
    #     datas = list()
    #     word_pairs =[['is', 'because']]
    #     for word in word_pairs:
    #         pattern = re.compile(r'\s?(is?)/[IN|VBZ]+\s(.*)(because|cause)/[IN|VBZ]+\s(.*)')
    #         result = pattern.findall(sentence)
    #         data = dict()
    #         if result:
    #             data['tag'] = result[0][0] + '-' + result[0][2]
    #             data['cause'] = result[0][3]
    #             data['effect'] = result[0][1]
    #             datas.append(data)
    #     if datas:
    #         return datas[0]
    #     else:
    #         return {}

    '''0由果溯因居端式精确'''
    def ruler0(self, sentence):
        '''
        cons3:因为、由于
        cons3_model:{Effect}<Conj...>{Cause}
        '''
        pattern = re.compile(r'(.*)[is|was|are|were|been]?[/VBD|a]?\s(caused.*\sby|arise.*\sfrom|arise.*\sout.*\sof|triggered.*\sby|induced.*\sby|the.*\scause.*\sof|affected.*\sby|effect.*\son)/[IN|c]+\s(.*)')
        result = pattern.findall(sentence)
        data = dict()
        if result:
            data['tag'] = result[0][1]
            data['cause'] = result[0][2]
            data['effect'] = result[0][0]

        return data

    '''1由果溯因配套式'''
    def ruler1(self, sentence):
        '''
        conm2: <cause of,is>, <>                     〈[之]所以,因为〉、〈[之]所以,由于〉、 <[之]所以,缘于〉
        conm2_model:<Conj>{Effect},<Conj>{Cause}
        '''
        datas = list()
        word_pairs =[['cause.*\sof', 'is'], ['cause.*\sof', 'was'], ['cause.*\sof', 'are'], ['cause.*\sof', 'were'], ['cause.*\sof', 'been'],
                     ['reason.*\sof','is'],['reason.*\sof','was'],['reason.*\sof','were'],['reason.*\sof','are'],['reason.*\sof', 'been'],
                     ['by.*\sreason.*\sthat','is'],['by.*\sreason.*\sthat','was'],['by.*\sreason.*\sthat','were'],['by.*\sreason.*\sthat','are'],['by.*\sreason.*\sthat', 'been'],
                     ['reason.*\swhy','is'],['reason.*\swhy','was'],['reason.*\swhy','are'],['reason.*\swhy','were'], ['reason.*\swhy', 'been'],
                     ['reason.*\sfor','is'],['reason.*\sfor','are'],['reason.*\sfor','was'],['reason.*\sfor','were'],['reason.*\sfor', 'were']]
        for word in word_pairs:
            pattern = re.compile(r'\s?(%s)/[IN|WRB]+\s(.*)(%s)/[VBZ|VBD|VBP]+\s(.*)' % (word[0], word[1]))
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
        conm1:
        conm1_model:<Conj>{Cause}, <Conj>{Effect}
        '''
        datas = list()
        word_pairs =[['is', 'cause.*\sfor'], ['was', 'cause.*\sfor'], ['are', 'cause.*\sfor'],["were","cause.*\sfor"],["been","cause.*\sfor"],
                    ['is', 'trigger.*\sof'], ['was', 'trigger.*\sof'], ['are', 'trigger.*\sof'],['were','trigger.*\sof'],['been','trigger.*\sof'],
                    ['if', 'then']]

        for word in word_pairs:
            pattern = re.compile(r'(.*)\s+(%s)/[VBD|VBZ|VBP|VBN|IN]+\s.*(%s)/[IN|RB]+\s(.*)' % (word[0], word[1]))
            result = pattern.findall(sentence)
            data = dict()
            if result:
                data['tag'] = result[0][1] + '-' + result[0][2]
                data['cause'] = result[0][0]
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
        cons2_model:{Cause},.<Conj...>{Effect}
        '''

        pattern = re.compile(r'(.*)[,，.。]*.*(hence|therefore,?|thus|thereby|accordingly,?|consequently,?|so|in this way|that\'s why|以至于?|从而|因而)/[RB|NN|NNP|IN]+\s(.*)',re.IGNORECASE)
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
        verb1: cause,            牵动、导向、使动、导致、勾起、引入、指引、使、予以、产生、促成、造成、引导、造就、促使、酿成、
        verb1_model:{Cause}(,)<Verb|Adverb...>{Effect}
        '''
        pattern = re.compile(r'(.*)\s+(causes?|causing|caused|triggers?|triggering|triggered|affects?|affecting|affected|induces?|inducing|induced|reveals?|revealed|revealing|leads? to|leading to|leaded to|brings? about|bringing about|brought about|brings? on|bringing on|brought on|gives? rise to|given rise to|giving rise to|increases?|increasing|increased|results? in|resulting in|resulted in|induces?|indeced|inducing|so.*\sthat.*\sto|have.*\seffect.*\son|has.*\seffect.*\son|had.*\seffect.*\son|having.*\seffect.*\son)/[VBG|VBP|VBD|VBN|VBZ|TO]+\s(.*)')
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
        pattern = re.compile(r'\s?(in.*\sorder.*\sto|for.*\sthe.*\spurpose.*\sof|after)/[IN|TO]+\s(.*)[,，]+(.*)',re.IGNORECASE)
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
        pattern = re.compile(r'\s?(because|as.*\sa.*\sresult.*\sof|due.*\sto|owning.*\sto|in.*\sview.*\sof|as.*\sa.*\sconsequence.*\sof|on.*\saccount.*\sof)/[IN|TO]+\s(.*)[,，]+(.*)', re.IGNORECASE)
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

        pattern = re.compile(r'(.*)(derive.*\sfrom|as.*\slong.*\sas|for.*\sthis.*\sreason.*\sthat|for.*\sthe.*\sreason.*\sthat|because.*\sof|since|for|through|after|stem.*\sfrom|result.*\sfrom|thanks.*\sto|as.*\sa.*\sresult.*\sof|due.*\sto|owning.*\sto|in.*\sconsequence.*\sof|in.*\sview.*\sof|as.*\sa.*\sconsequence.*\sof|on.*\saccount.*\sof|in.*\sthat|on.*\sthe.*\sgrounds.*\sthat|as)/[IN|TO|DT]+\s(.*)')
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
        if self.ruler0(sentence):
            print(0)
            infos.append(self.ruler0(sentence))
        elif self.ruler1(sentence):
            print(1)
            infos.append(self.ruler1(sentence))
        elif self.ruler2(sentence):
            print(2)
            infos.append(self.ruler2(sentence))
        elif self.ruler3(sentence):
            print(3)
            infos.append(self.ruler3(sentence))
        elif self.ruler4(sentence):
            print(4)
            infos.append(self.ruler4(sentence))
        elif self.ruler5(sentence):
            print(5)
            infos.append(self.ruler5(sentence))
        elif self.ruler6(sentence):
            print(6)
            infos.append(self.ruler6(sentence))
        elif self.ruler7(sentence):
            print(7)
            infos.append(self.ruler7(sentence))
        elif self.ruler8(sentence):
            print(8)
            infos.append(self.ruler4(sentence))
        # elif self.ruler9(sentence):
        #     infos.append(self.ruler9(sentence))

        return infos

    '''抽取主控函数'''
    def extract_main(self, content):
        sentences = self.splitSentence(content)
        datas = list()
        for sentence in sentences:
            subsents = self.fined_sentence(sentence)
            subsents.append(sentence)
            for sent in subsents:
                sent=sent.replace('\n','')
                sent=sent.strip()
                text =sent.split()
                tokens_tag = pos_tag(text)
                sent = ' '.join([word[0] + '/' + word[1] for word in tokens_tag])
                # sent = ' '.join([word.word + '/' + word.flag for word in pseg.cut(sent)])
                result = self.extract_triples(sent)
                if result:
                    for data in result:
                        if data and  data['tag'] and data['cause'] and data['effect']:
                            datas.append(data)
        return datas

    '''文章分句处理'''
    # def process_content(self, content):
    #     return [sentence for sentence in SentenceSplitter.split(content) if sentence]

    '''SplitSentences'''
    def splitSentence(self,paragraph):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(paragraph)
        return sentences

    '''切分最小句'''
    def fined_sentence(self, sentence):
        return re.split(r'[?!,;]', sentence)


'''测试'''
def test():
    content0 = """他之所以这么嚣张，是因为我们太软弱了"""
    content0_en = """He is so arrogant, because we are too weak. shit!"""

    content_rule_8_base='''
      He died as long as a noble cause.
      He died due to a noble cause.
      He died owning to a noble cause.
      He died for a noble cause.
      He died derive from a noble cause.
      He died because of a noble cause.
      He died since a noble cause.
      He died through killing himself.
      He died after a noble cause.
      He died stem from a noble cause.
      He died result from a noble cause.
      He died thanks to a noble cause.
      He died through killing himself.
      He died after a noble cause.
      He died stem from a noble cause.
      He died result from a noble cause.
      He died thanks to a noble cause.
      He died as a result of a noble cause.
      He died in consequence of a noble cause.
      He died in view of a noble cause.
      He died as a consequence of a noble cause.
      He died on account of a noble cause.
      He died in that a noble cause.
      He died on the grounds that a noble cause.
      He died as a noble cause.
      He died for this reason that a noble cause.
      He died for the reason that a noble cause.
    '''

    content_rule_7_base='''
    As a result of their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    Because their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    Due to their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    owning to their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    in view of their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    as a consequence of their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    on account of their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    '''

    content_rule_6_base='''

    '''

    content_rule_5_base='''
    for the purpose of maximize profit, the firm would seek to maximize output.
    In order to maximize profit, the firm would seek to maximize output.
    after installing maximize profit, the firm would seek to maximize output.
    '''

    content_rule_4_base='''
    the dog causes the bug problem.
    For one thing, the future does not exist, so that to forge images of it is a kind of lie.
    the dog cause the bug problem.
    Dirt clogs the pores, causing blemishes.
    the dog caused the bug problem.
    This could trigger a memory about what you're talking about through that lecture, which can then trigger another memory.
    This triggered a memory about what you're talking about through that lecture, which can then trigger another memory.
    This is triggering a memory about what you're talking about through that lecture, which can then trigger another memory.
    Does television affect children's behaviour?
    Does television affected children's behaviour?
    Does television be affecting children's behaviour?
    Whales being sociable animals probably need the stimulus of sizeable gatherings to induce reproductives behavior.
    The afternoon drifted by heavily, inducing sleep.
    The afternoon drifted by heavily induced sleep.
    The Committee released the minutes of its last meeting, revealing the group that sets interest rates thought unemployment could rise to 10% this year.
    The report revealed a great deal of bureaucratic inefficiency.
    The report reveals a great deal of bureaucratic inefficiency.
    Curiosity can also be dangerous, leading to setbacks or even downfalls.
    The curiosity leads to setbacks or even downfalls.
    The curiosity leaded to setbacks or even downfalls.
    Putting pressure on the country bring about political change.
    Putting pressure on the country brought about political change.
    Putting pressure on the country bringing about political change.
    Putting pressure on the country bring on political change.
    Putting pressure on the country brought on political change.
    Putting pressure on the country bringing on political change.
    Uniforms also give rise to some practical problems.
    Uniforms also given rise to some practical problems.
    Uniforms are also giving rise to some practical problems.
    Reading will increase your vocabulary.
    Reading increased your vocabulary.
    Reading is increasing your vocabulary.
    Such a war could result in the use of chemical and biological weapons
    Such a war resulted in the use of chemical and biological weapons
    Such a war is resulting in the use of chemical and biological weapons
    Music can induce a meditative state.
    Music is inducing a meditative state.
    Music induced a meditative state.
    '''

    content_rule_3_base='''
    The cost of materials rose sharply last year, Accordingly, we were forced to increase our prices.
    The cost of materials rose sharply last year, Accordingly we were forced to increase our prices.
    Here is so huge, consequently even this vast chorus was occasionally overwhelmed by the brass.
    We suspect they are trying to hide something, so the need for an independent inquiry.
    In this way, it learned to paint.
    He reads the texts every morning. In this way, he is able to recite them.
    Everybody makes mistakes, that's why they put erasers on pencils.
    We suspect they are trying to hide something, hence the need for an independent inquiry.
    He's only 17 and therefore not eligible to vote.
    We suspect they are trying to hide something, thus the need for an independent inquiry.
    Regular exercise strengthens the heart, thereby reducing the risk of heart attack.
    They had children and were consequently tied to the school vacations.
    '''

                    # ['if', 'then']]
    content_rule_2_base='''
    Her triumph was a cause for celebration.
    Her triumph is a cause for celebration.
    The doctor said there was no cause for alarm.
    Toys were a cause for celebration.
    Toys are a cause for celebration.
    if do A then he do C.
    '''

    content_rule_1_base='''
    The cause of the fire is the wood.
    The cause of the fire was the wood.
    The cause of the fire were the wood.
    The cause of the fire are the wood.
    The real reason of course is laziness.
    The real reason of course are laziness.
    The real reason of course was laziness.
    The real reason of course were laziness.
    So the people asked him to hold the candle for Don Twelfth, just by reason that his beautiful was more glaringness than the luna.
    So the people asked him to hold the candle for Don Twelfth, just by reason that his beautiful were more glaringness than the luna.
    So the people asked him to hold the candle for Don Twelfth, just by reason that his beautiful is more glaringness than the luna.
    So the people asked him to hold the candle for Don Twelfth, just by reason that his beautiful are more glaringness than the luna.
    The reason why the injection needs repeating every year is that the virus changes.
    My sole reason for coming here is to see you.
    My sole reason for coming here was to see you.
    My sole reason for coming here are to see you.
    My sole reason for coming here were to see you.
    '''



    content_rule_0_base='''
    HyperStack can use default credentials to connect to IPC$ shares on remote machines.[5]
    The accident was caused by pilot error.
    Noisome vapours arise from the mud left in the docks.
    A contract cannot arise out of an illegal act.
    The laugh at that time are triggered by surprise in a safe situation (think peek-a-boo), and don't just endear babies to their parents.
    Opacity of the eye lens can be induced by deficiency of certain vitamins.
    We have achieved great successes in the cause of building up our country.
    From my point of view, TV is the cause of the declining interest in school and the failure of our entire educational system.
    We are profoundly affected by what happens to us in childhood.
    A word from the teacher will have a great effect on my son.
    '''





    content_cause = """
    He died for a noble cause.
    The drugs tend to cause drowsiness.
    The cause of the fire is not yet known.
    The immediate cause of death is unknown.
    Unemployment is a major cause of poverty.
    Tree roots can cause damage to buildings.
    High temperatures can cause hallucination.
    Idealistic young people died for the cause.
    The photo may cause offence to some people.
    Unemployment was the chief cause of poverty.
    She was the unwitting cause of the argument.
    This would cause inflation to let rip again.
    Caffeine can cause palpitations and headaches.
    The virus can cause pregnant animals to abort.
    She was soon converted to the socialist cause.
    Cars cause pollution, both smog and acid rain.
    It had been the cause of much emotional upset.
    Eating too much salt can cause fluid retention.
    Faulty wiring is the major cause of house fires.
    Vitamin deficiency in the diet can cause illness.
    The disease can cause sterility in men and women.
    These gases cause untold damage to the environment.
    The drug may cause an aggravation of the condition.
    Heavy seas can cause cancellation of ferry services.
    Brushing wet hair can cause stretching and breakage.
    Her inability to concentrate could cause an accident.
    Reports of this kind are guaranteed to cause anxiety.
    He was the unknowing cause of all the misunderstanding.
    They worked together to advance the cause of democracy.
    Heavy drinking can cause permanent damage to the brain.
    """
    content_arise = '''
    Have a plan of action To combat any type of sibling conflict that might arise, experts suggest tons of communication among the family, as well as a plan of action.
    In the end, this will aid the company in potentially avoiding lawsuits that otherwise might arise if the AI is being produced that is not abiding by AI Ethics precepts.
    Just beware of any fees or taxes that might arise with each transaction.
    Discrepancies between the trial data and real-world experience might arise from the timing of the original research.
    But some community members have raised questions about the integrity of the canisters and the plans for addressing any problems that might arise.
    Regardless of what internal drama or setbacks might arise, the Warriors can expect their no-frills big man to arrive early, stay late and do the little things — screening, boxing out, executing dribble-handoffs — required of championship runs.
    The rats didn’t show signs of health issues like seizures or epilepsy, which the researchers had worried might arise.
    These bonds cover any liabilities that might arise if an agricultural producer fails to honor its contract with migrant workers and is a way to protect those workers from abusive employment conditions.
    '''
    content_trigger = '''
    He kept his finger on the trigger. 
    Stress may act as a trigger for these illnesses. 
    Nuts can trigger off a violent allergic reaction. 
    A man pointed a gun at them and pulled the trigger. 
    Unresolved or unacknowledged fears can trigger sleepwalking. 
    The trigger for the strike was the closure of yet another factory. 
    Sometimes even a light touch on the face is enough to trigger off this pain. 
    Wire the thermometers up to trigger off an alarm bell if the temperature drops. 
    There are other trigger points. 
    When you click the trigger, it pops up an alert. 
    Does the thawing process has some kind of trigger as well? 
    Why does social media trigger feelings of loneliness and inadequacy? 
    This could trigger a tsunami that would create havoc along the coast. 
    With these functions, we can find out everything about the trigger or its context. 
    The photograph is able to achieve this result and is able to trigger these thoughts. 
    Even when they do trigger lightning, things still do not always go according to plan. 
    The problem with novelty, however, is that it tends to trigger the brain's fear system. 
    One cognitive theory suggests that aggravating and painful events trigger unpleasant feelings. 
    This could trigger a memory about what you're talking about through that lecture, which can then trigger another memory. 
    The state already has a "trigger law" on the books, which would ban most abortions, if the Supreme Court overturn Roe v Wade. 
    Thunder, the shock wave that comes from a lightning flash, is thought to be the trigger for the torrential rain that is typical of storms. 
    If you squeeze the trigger of a gun really hard and really fast, it doesn't fire any faster or harder than if you just squeezed it gently. 
    Some of them are a little trigger-happy—they'll shoot at anything that moves. 
    A hair-trigger situation has been created which could lead to an outbreak of war at any time. 
    His boozing, arrogance, and hair-trigger temper have often led him into ugly nightclub brawls. 
    It is argued, stripping cartons of their branding will trigger no mass movement to quit. 
    For example, we can change how often we receive the distracting notifications that trigger our urge to check. 
    The findings contradict research published earlier this year showing that returning adult trigger a significant decline in their parents' quality of life and wellbeing. 
    '''
    content_increase= '''
    Prices will increase pro rata. 
    We need to increase productivity. 
    Car usage is predicted to increase. 
    The population continues to increase. 
    Reading will increase your vocabulary. 
    The report recommended a 10% pay increase. 
    Worn rugs increase the danger of tripping. 
    Our main aim is to increase sales in Europe. 
    Short-term contracts increase staff turnover. 
    There was an astounding 20% increase in sales. 
    We needed to increase the volume of production. 
    Obesity can increase the risk of heart disease. 
    Wayne plans to increase the print run to 1,000. 
    This sort of increase simply cannot be justified. 
    That would increase Olympia & York's holding to 35%. 
    A good advertising campaign will increase our sales. 
    The food had been adulterated to increase its weight. 
    Companies need to take active steps to increase exports. 
    All parties are promising to increase spending on health. 
    Smoking can increase the risk of developing heart disease. 
    The tax increase sounded the death knell for the business. 
    There has been a marked increase in crimes against property. 
    The directors have just voted themselves a huge pay increase. 
    Very vigorous exercise can increase the risk of heart attacks. 
    It would be highly undesirable to increase class sizes further. 
    We are now witnessing an unprecedented increase in violent crime. 
    Testosterone does not increase their erectile or orgasmic ability. 
    Cases of breathing difficulties increase in lockstep with air pollution. 
    Mugging is on the increase. 
    We've been given a 2% pay increase. 
    '''

    content_affect='''
    How will these changes affect us? 
    Your opinion will not affect my decision. 
    Does television affect children's behaviour? 
    The new proposals affect both clergy and laity. 
    This may affect your entitlement to compensation. 
    The drug may affect your powers of concentration. 
    Climate and weather affect every aspect of our lives. 
    Cataracts affect the transparency of the eye's lenses. 
    Many external influences can affect your state of mind. 
    The new law will affect us all, directly or indirectly. 
    Your contributions will affect your pension entitlements. 
    An unhappy home environment can affect children's behaviour. 
    You never allow personal problems to affect your performance. 
    Irritable bowel syndrome seems to affect more women than men. 
    People tend to think that the problem will never affect them. 
    You're fooling yourself if you think none of this will affect you. 
    Every ingestion of food can affect our mood or thinking processes. 
    This will negatively affect the result over the first half of the year. 
    He is interested in how our perceptions of death affect the way we live. 
    Price changes must not adversely affect the living standards of the people. 
    The warming of the Earth and the consequent climatic changes affect us all. 
    Even in the artificial environment of an office, our body rhythms continue to affect us. 
    Changes in the money supply affect the level of economic activity and the interest rate. 
    His words have great affect on me. 
    How do daydreams affect daydreamers? 
    Intuition may affect reflective tasks. 
    I don't smoke, so it doesn't affect me. 
    When it grows hot, heat can affect human health. 
    Individual sins affect the entire fabric of society. 
    '''

    content_effect='''
    We warned them, but to no effect. 
    Her words had a magical effect on us. 
    His voice had an almost hypnotic effect. 
    The effect has proved hard to reproduce. 
    His words had exactly the opposite effect. 
    The crisis had a negative effect on trade. 
    The effect is almost impossible to describe. 
    The medicine did not achieve the desired effect. 
    Affairs do have a devastating effect on marriages. 
    My father's death had a profound effect on us all. 
    The overall effect of the painting is overwhelming. 
    She is unconscious of the effect she has on people. 
    Does television have an effect on children's behaviour? 
    The announcement had a dramatic effect on house prices. 
    The stage lighting gives the effect of a moonlit scene. 
    The calming effect seemed to last for about ten minutes. 
    They had seriously miscalculated the effect of inflation. 
    Unemployment is having a corrosive effect on our economy. 
    Overseas investment has had a positive effect on exports. 
    The price increase has had no perceptible effect on sales. 
    This effect is particularly noticeable in younger patients. 
    The specific impact of the greenhouse effect is unknowable. 
    Petty crime is having a deleterious effect on community life. 
    His mother's untimely death had a catastrophic effect on him.
    The aspirins soon take effect. 
    They hope to effect a reconciliation. 
    The colour green has a restful effect. 
    The new law takes effect from tomorrow. 
    One side effect of modern life is stress. 
    New controls come into effect next month. 
    '''

    content_induce='''
    Nothing would induce me to take the job. 
    Whales being sociable animals probably need the stimulus of sizeable gatherings to induce reproductives behavior. 
    Indeed, the mere presence of a grape in the other chamber (without an actual monkey to eat it) was enough to induce resentment in a female capuchin. 
    In view of this behavior it has been suggested that chemicals present in fresh buck rubs may help physiologically induce and synchronize fertility in females that visit these rubs. 
    We'll have to induce her. 
    Music can induce a meditative state. 
    Doctors said surgery could induce a heart attack. 
    Taking a brisk walk can often induce a feeling of well-being. 
    Like fireplaces, they induce sense of comfort and warmth. 
    For humans, hearing a sudden loud noise might prove frightening, but it does not induce mass fatality. 
    It's all about making the traveling experience less stressful and blue is said to induce a feeling of calm. 
    At 15 euro ($22) a tonne the price is high enough to induce power companies to switch some generation from coal to ga. 
    When people engage in activities that help others, their brain releases endorphins, the brain's natural opiates, which induce in people a feeling of well-being. 
    The authors explore factors that induce Veblen effects. 
    We also suggest that pimobendan may induce ventricular hypertrophy. 
    Induce Giardia lamblia C2 strain to complete its life cycle in vitro. 
    Even modest doses of narcotine can induce profound nausea and vomiting. 
    Xuefen syndromes of epidemic febrile disease was made in rabbits by endotoxin to induce generalized Shwartzman reaction. 
    Only a few other drugs, such as methohexital, etomidate, or propofol have the capability to induce anesthesia so rapidly. 
    Many causes could induce rhabdomyolysis, such as exercise, thermoplegia, or autoallergic disease. 
    Herbal teas can help induce sleep. 
    But it can also induce nervousness. 
    I don't say this to induce a guilt trip. 
    Presumably, all these taxes are to induce compliance. 
    The castor bean can also induce labor in pregnant women. 
    These taxonomies induce semantic relationships among objects. 
    Meeting new people is fun. But it can also induce nervousness. 
    And corroding anxiety and bad temper actual induce physical diseases. 
    Listening to soothing music can calm the senses and help induce sleep.
    '''

    content_derive='''  
    In fact, over fifty per cent of genuine British surnames derive from place names of different kinds, and so they belong to the last of our four main categories. 
    Most patients derive enjoyment from leafing through old picture albums. 
    Mr. Ying is one of those happy people who derive pleasure from helping others. 
    They derive greater pleasure from buying things. 
    Elegance, he believed, did not derive from abundance. 
    Only rarely can we derive any "real" quantities from deposits of broken pots. 
    The acquisition acts of bidders may derive from modeling: a manager does what other managers do. 
    It should be noted that in these cases, a reader can derive the intended meaning from the context. 
    It's possible that the birds can derive so much energy from these grubs that they only need to eat a few each day. 
    In one area, it made painstaking efforts to quantify fertility preference to derive figures for planned and unplanned pregnancies. 
    When food is in short supply, a ruminant can last longer than a nonruminant because it can derive more energy out of the same food. 
    These all revealed that the DNA profile found inside the gourd is extremely rare in modern Eurasians, suggesting that it may derive from a royal bloodline. 
    Although pressure to recruit women directors, unlike that to employ women in the general work force, does not derive from legislation, it is nevertheless real.
    At the same time, we derive massive economic benefits when we buy the most affordable energy on the world market and when we engage in energy trade around the world. 
    A child's ability to become deeply absorbed in something, and derive intense pleasure from that absorption, is something adults spend the rest of their lives trying to return to. 
    Student tutors feel upset when their teachable students fail, but happy when these virtual pupils succeed as they derive pride and satisfaction from someone else's accomplishment. 
    Late endosomes might derive from early endosomes by stepwise maturation. 
    Now you can derive this. 
    Many later advice manuals derive from it. 
    Derive new knowledge. 
    But I'm never going to ask you to derive things. 
    You cannot derive a complete architecture with TDD. 
    You want people who derive joy from the work they do.
    But the local Dinka derive no benefit from this money. 
    And continue to derive the macroscopic entropy changes. 
    You need to derive a filename to store this information. 
    Organizations derive greater value from existing content. 
    '''

    content_reveal='''
    These poems reveal her gentle side. 
    Officers could not reveal how he died. 
    She has refused to reveal the whereabouts of her daughter. 
    Beware of saying anything that might reveal where you live. 
    The report was a device used to hide rather than reveal problems. 
    We reveal only as much information as we can safely risk at a given time. 
    To get you started, we have asked five successful writers to reveal some of the tricks of the trade. 
    What do our responses reveal about us? 
    Reveal your thoughts and express your ideas to yourself. 
    Most people surveyed were reluctant to reveal what they ate. 
    Its plots and events reveal a lot about Frankie's actual life. 
    That would reveal which ones we have seen and which we have not. 
    Images of the Martian surface reveal many hundreds of volcanoes. 
    In some cases, a simple Google search can reveal what you think. 
    Spectroscopy can reveal the composition of those touch-up layers too. 
    Her portraits reveal her own interpretation of her subject's state of mind. 
    Companies could be forced to reveal to consumers what information they hold. 
    Exposition is whatever background information you have to reveal to the audience. 
    After the two servants reveal all this background information, we meet the young man. 
    Longevity statistics reveal that the average person doesn't last very long after retirement. 
    In a study published in Nature Scientific Reports, we reveal just how deep this injustice runs. 
    No one has found one of their canoes or any rigging, which could reveal how the canoes were sailed. 
    Such experiments cannot, however, reveal whether the birds were reexperiencing the past when retrieving the information. 
    Subsequently developed methods of recording and analyzing nerve potentials failed to reveal any such qualitative diversity. 
    The flower opens to reveal a bee. 
    She doesn't reveal much of her inner self. 
    The door opened to reveal a cosy little room. 
    His writings reveal an unattractive preciousness of style. 
    She crouched in the dark, too frightened to reveal herself. 
    Companies should be made to reveal more about their financial position. 
    '''

    content_because='''
    They are here because of us. 
    She resigned because of ill health. 
    I was angry because I played so badly. 
    The road is closed because of the snow. 
    Many diets fail because they are boring. 
    He walked slowly because of his bad leg. 
    She's been off work because of sickness. 
    Feed the dogs because they haven't eaten. 
    They abandoned the match because of rain. 
    She had left him because of his drinking. 
    I started to cry because I cut my finger. 
    I didn't get it because it cost too much. 
    They knocked off $60 because of a scratch. 
    John liked him because he was a nervy guy.
    Some roads are closed because of drifting. 
    Filming was delayed because of bad weather. 
    He won't resign because he's not a quitter. 
    She had scratched because of a knee injury. 
    She was marked down because of poor grammar. 
    I agreed, but only because I was frightened. 
    I had to refuse because of a prior engagement. 
    Officially, he resigned because of bad health. 
    He was forced to retire because of ill health. 
    We had a humongous fight just because she left. 
    The game was called off because of bad weather. 
    I chose boxing because it is my favourite sport. 
    People learned self-reliance because they had to. 
    The goal was disallowed because Wark was offside. 
    Cassandra noticed him because he was good-looking. 
    '''

    content_so='''
    Some seed varieties germinate fast, so check every day or so.
    '''

    content_hence='''
    We suspect they are trying to hide something, hence the need for an independent inquiry.
    The trade imbalance is likely to rise again in 2007. Hence a new set of policy actions will be required soon.
    Hence, the Jovian planets are often called giants.
    Hence the growing importance of the student survey.
    Hence, the transmission-reception system breaks down.
    For centuries, medicine was impotent and hence unproblematic.
    Hence the description of America as a "graveyard" for languages.
    Smell is cultural, hence it is a social and historical phenomenon.
    Hence, 1930 and 1948 are generally considered bookends to Hollywood's Golden Age.
    Other Europeans wrongly thought them migrant Egyptians, hence the derivative Gypsy.
    Other Europeans (wrongly) thought them migrant Egyptians, hence the derivative Gypsy.
    Generally, the layering occurs on an annual basis, hence the observed changes in the records can be dated.
    Eighty percent of them are thought to involve small birds like doves and larks, and hence go largely unnoticed.
    It is a rather shallow bowl that was crudely made in a mold; hence, in only a limited number of standard sizes.
    It is good to consume tomatoes as they have lycopene, which is an antioxidant and hence works as a sunscreen from within.
    Admittedly, young people are generally more energetic than the elders, hence it is more likely for them to come up with new ideas.
    The descending rock is substantially cooler than the surrounding mantle and hence is less ductile and much more liable to fracture.
    Hence, there is a real concern throughout Europe about the damage to the forest environment which threatens these three basic roles.
    Carousels are essentially slideshow navigations, in which the content rotates vertically or horizontally (hence the name "carousel").
    All the isotopes of a given element have the same number of protons, but differ in their number of neutrons, and hence in their atomic mass.
    They can do only so much to represent the full complexity of the global climate and hence may give only limited information about natural variability.
    It had a wider and deeper hull than the galley and hence could carry more cargo: increased stability made it possible to add multiple masts and sails.
    Hence the analogy that likens the conduct of monetary policy to driving a car with a blackened windscreen, a cracked rear-view mirror and a faulty steering wheel.
    Hence, depictions of violence among teenagers should be prohibited from movies and television programs, if only in those programs and movies promoted to young audiences.
    If the population of these other species were increased, the number of ticks acquiring the bacterium and hence the number of people contracting Lyme disease would likely decline.
    Generally, young women have had growing success in the paid labor market since 1960 and hence might increasingly be expected to be able to afford to live independently of their parents.
    The true consequences will only be known several years hence.
    The gases that may be warming the planet will have their main effect many years hence.
    '''

    content_therefore='''
    He's only 17 and therefore not eligible to vote.
    They had seen him trawling and therefore knew that there were fish.
    Muscle cells need lots of fuel and therefore burn lots of calories.
    Environmental systems tend to be nonlinear, and therefore not easy to predict.
    Users therefore can search by title, by author, by subject, and often by keyword.
    Punishment cannot, therefore, be discussed in isolation from social and political theory.
    There is still much to discuss. We shall, therefore, return to this item at our next meeting.
    The flexibility of the lens decreases with age; it is therefore common for our sight to worsen as we get older.
    Therefore, few people wrote.
    Therefore, being careful is necessary.
    Their bodies have therefore become unhealthy.
    Therefore, additional explanations are needed.
    Take him, therefore, and throw him into prison.
    Therefore, we should all bear in mind his advice.
    Therefore, less money is spent for garbage disposal.
    Therefore, he can go shopping without being recognized.
    This seems, therefore, to be an approach worth exploring.
    They had therefore to go out and look for the second time.
    The greatest water storage, therefore, lies near the surface.
    The best observatory sites are therefore high, dark, and dry.
    Variations in wood density affect vibrations, and therefore, sound.
    Therefore, cannabinoids probably function to stimulate the appetite.
    The existence of meeting facilities therefore seems high at airports.
    There were three, therefore, that did not go; the others hastened on.
    To some extent, therefore, the principle of secrecy had been maintained.
    They will die, therefore, not with that familiar "Ping! " and "Oh, drat!"
    The judge, therefore, reversed, at least partially, his original decision.
    '''

    content_thus='''
    Many scholars have argued thus.
    He is the eldest son and thus heir to the title.
    Jimmy is adopted and thus unrelated to Beth by blood.
    He thus avoided a pack of journalists eager to question him.
    Scientific materialism thus triumphed over ignorance and superstition.
    As we age, the lenses of the eyes thicken, and thus refract light differently.
    Sex education is a sensitive area for some parents, and thus it should remain optional.
    The universities have expanded, thus allowing many more people the chance of higher education.
    Neither of them thought of turning on the news. Thus Caroline didn't hear of John's death until Peter telephoned.
    Thus the correct answer is dictate.
    Government thus needs to act as well.
    Thus the first worker's league came into being.
    Thus the first workers' league came into being.
    Thus, he could learn more new words and grammar.
    It is only thus that any one may sight those magic shores.
    The vicious circle is thus transformed into a virtuous circle.
    Thus, rising microwave sales could be seen as a positive thing.
    Tom chased the traitor home, and thus found out where he lived.
    They cried, for it was always thus that he signalled his return.
    They are difficult to maintain and thus need more operating costs.
    Thus, papa insisted that we learn at least one new thing each day.
    Thus, natural selection stabilized this host-parasite relationship.
    Thus species selection is actually a result of individual selection.
    Thus I have been allowed to gain access to their living environment.
    Thus, how we see ourselves reflects the views of us that others communicate.
    Fossils are thus identifying markers for particular periods in Earth's history.
    An "aristocracy of virtue and talent" thus could be recruited from all classes.
    Thus, our newer employees are not generating enough revenue to justify their salaries.
    She misinterpreted the implications of his letter and thus misunderstood his intentions.
    '''

    content_thereby='''
    Regular exercise strengthens the heart, thereby reducing the risk of heart attack.
    This had been dismantled in 1933, thereby breaking the link.
    This gives you the opportunity to pray for and bless the other, thereby blessing yourself.
    He thereby transformed an inefficient pump of limited use into a steam engine of a thousand uses.
    This thereby leads them to expect the same kind of spoilt treatment throughout their adult lives.
    Reduced operation of the machinery should limit their production and thereby constrain the damage.
    When an object is encountered again, it is matched with its internal representation and thereby recognized.
    In the winter, ECC can be heated by passing an electric current through it, thereby preventing ice buildup.
    It carries fine particles of sand, which bombard exposed rock surfaces, thereby wearing them into yet more sand.
    They are thereby shut off from the world of books and newspapers, having to rely on friends to read aloud to them.
    Retailers that master the intricacies of wholesaling in Europe may well expect to rake in substantial profits thereby.
    Retailers that master the intricacies of wholesaling in Europe may well expect to rake in substantial profits thereby.
    This may have deformed the lower ice, disrupting its annual layers and thereby causing the discrepancy between the records.
    It is thought that these grooves help to channel water through the mouth and out the gill slits, thereby reducing water resistance.
    They can push down the thirteenth partial and bring up the fifteenth and thereby change the sound of a clarinet into a French horn.
    When a large body strikes a planet or moon, material is ejected, thereby creating a hole in the planet and a local deficit of mass.
    Dependency injection lets you weave together the main layers of your application, thereby lets you produce a loosely coupled application.
    This research may be biased, however, as ill health often makes older people more dependent and thereby increases contact with family members.
    This research may be biased, however, as ill-health often makes older people more dependent and thereby increases contact with family members.
    At the same time, they would break the illegal excavator's grip on the market, thereby decreasing the inducement to engage in illegal activities.
    Since nicotine in cigarettes changes brain chemistry, perhaps thereby affecting mood, it is likely that smoking contributes to depression in teenagers.
    In contrast, the roots of grasses and other small plants may help to hold loose soil fragments together, thereby helping to prevent erosion by the wind.
    The social network will let users "mute" messages from other users on their timelines without the muted person's knowledge, thereby avoiding the awkward.
    They would have prevented the reseeding of the slow-growing palm trees and thereby doomed Rapa Nui's forest, even without the settlers' campaign of deforestation.
    Still, if late sleepers want to lose a few pounds, they can go to bed earlier than they usually do, thereby reducing their chances of taking snacks before bedtime.
    Still, if late sleepers want to lose a few pounds, they can go to bed earlier than they usually do, thereby, reducing their chances of taking snacks before bedtime.
    Stress reactions can reduce the disease fighting effectiveness of the body's immune system, thereby increasing susceptibility to illnesses ranging from colds to cancer.
    The sparse distribution of the feathers, however, also allows considerable lateral air movement over the skin surface, thereby permitting further heat loss by convection.
    In 1990, in an attempt to reduce alcohol consumption and thereby to reduce alcohol-related traffic deaths among Berinians under 21, the legal drinking age was raised to 21.
    One option for moving toward both biodiversity and terrestrial food supply goals is to produce greater yields from less land, thereby freeing land for conservation purposes.
    '''

    content_since='''
    Since this method doesn’t work, let’s try another.
    '''

    content_for='''
    Let's hear it for the teachers, for a change.
    They call me the gofer—go for this, go for that...
    She fought honestly for a just cause and for freedom.
    Tullio has been modelling for Sandra for eleven years.
    For the time being it's business as usual for consumers.
    They had liquor for the adults and sodas for the children.
    There is potential for selective breeding for better yields.
    He achieved fame for his stage sets for the Folies Bergeres.
    Farmers struggling for survival strip the forests for agricultural land.
    He was jailed for life for murder.
    What's for breakfast?
    I'm not afraid for me, but for the baby.
    You were as hot for me as I was for you.
    What's for dessert ?
    What's for pudding?
    What's for supper?
    I've sent off for some books for my course.
    He could be jailed for two years for contempt.
    We've been shooting for a pay raise for months.
    We had to queue up for an hour for the tickets.
    The area has been mined for slate for centuries.
    The pilot was waiting for clearance for take-off.
    For three years, I have probed for understanding.
    These items are just for show—they're not for sale.
    He talked for two hours without pausing for breath.
    He spied for his government for more than ten years.
    She's searching for subject matter for her new book.
    He was jailed for two years for fraud and deception.
    The kids were scouting around for wood for the fire.
    Marks were noticeably higher for girls than for boys.
    '''

    content_through='''
    He breezed through the tests.
    I looked through the keyhole.
    We zoomed through the gallery.
    He walked through customs.
    'Tough' and 'through' don't rhyme.
    Are you through with this?
    They got the bill through Congress.
    They marched silently through the streets.
    Most advertisements work through suggestion.
    I climbed through the window.
    We'll muddle through somehow.
    She whizzed through the work.
    I sorted through my paperwork.
    She glanced through the report.
    We drove through hail and snow.
    Applause rang through the hall.
    He powered through the water.
    She linked her arm through his.
    Water gurgled through the pipes.
    The canoe cut through the water.
    The deal did not go through.
    He's British through and through.
    Her goodness shone through.
    The sun burst through the clouds.
    A message is just coming through.
    The operator will put you through.
    Relief surged through her.
    We're through with dinner.
    Children scavenge through rubbish.
    The sun slanted through the window.
    '''

    content_after='''
    Not long after that he resigned.
    She was paroled after two years.
    He did push-ups after games.
    "Come back!" he called after me.
    I'll do my homework after supper.
    The class reassembled after lunch.
    I am a seeker after truth.
    After the revolution, anarchy ruled.
    We got off straight after breakfast.
    She died shortly after giving birth.
    We'll leave after lunch.
    We sat about languidly after dinner.
    Everything changed after nine-eleven.
    She's leaving the day after tomorrow.
    There was euphoria after the election.
    Play resumed quickly after the stoppage.
    He's mending slowly after the operation.
    Sadly, bamboo plants die after flowering.
    After intermission, the second band played.
    Reapply sunscreen hourly and after swimming.
    She developed complications after the surgery.
    She arrived shortly after us.
    They arrived shortly after 5.
    She was left staring after him.
    She died after a long illness.
    He died after a long illness.
    He's the tallest, after Richard.
    Regulars return year after year.
    The sky cleared after the storm.
    He went after the burglars.
    '''

    content_accordingly='''
    It is a difficult job and they should be paid accordingly.
    The cost of materials rose sharply last year. Accordingly, we were forced to increase our prices.
    We have a different background, a different history. Accordingly, we have the right to different futures.
    They realized that some of their prices were higher than their competitors' and revised prices accordingly.
    They adjust their goals accordingly.
    Accordingly, chalkbrood is most common in spring and in small colonies.
    Accordingly, newspapers were read almost only by rich people in politics or the trades.
    Accordingly, high achievers should not be put into jobs that are inconsistent with their needs.
    Accordingly, green space is thought to prompt important exercises in discovery, creativity, and risk-taking.
    Accordingly, managing your job search properly is just as important as identifying job opportunities and submitting your applications.
    The neurons cultivated at Johns Hopkins—unlike those culled from tumors are not cancerous, and accordingly are much more valuable as research tools.
    At the same time these computers record which hours are busiest and which employees are the most efficient, allowing personnel and staffing assignments to be made accordingly.
    The Kyoto Protocol, which set different duties for all countries to cut emissions, states that different countries should reduce their collective greenhouse emissions accordingly.
    In contrast, high-frequency fluctuations such as seasonal temperature variations are observable and somewhat predictable, so that groups could have adapted their behaviors accordingly.
    We have to discover his plans and act accordingly.
    Adapt your style accordingly to type.
    The meanings we associate with the phrase will change accordingly.
    They went and foraged accordingly, hunting through every cupboard and turning out every drawer.
    The Prince accordingly made her his wife; being now convinced that he had found a real Princess.
    The importance of non-financial firms will accordingly rise, along with their ability to attract the best talent.
    Carpenter supervisors oversee carpentry work on a specified project to ensure that workers are on schedule and executing plans accordingly.
    When corn came to Europe from Mexico, now they had a much hardier crop that could be grown easily in more northerly climates and the centers of power began to shift accordingly.
    See that you act accordingly.
    Please note and take appropriate action.; Please consider and act accordingly.
    The market has danced accordingly.
    And he buys life insurance accordingly.
    Some of the books suffer accordingly.
    The client logic is developed accordingly.
    Demand for water will rise accordingly.
    '''

    content_consequencely='''
    They had children and were consequently tied to the school vacations.
    Consequently even this vast chorus was occasionally overwhelmed by the brass.
    Consequently, even as the blogosphere continues to expand, only a few blogs are likely to emerge as focal points.
    Grandfather had sustained a broken back while working in the mines. Consequently, he spent the rest of his life in a wheelchair.
    Consequently, government revenue from the tax will increase.
    In college, time is scarce, and consequently, very precious.
    Consequently, a black market in imported illicit CFC's has grown.
    Consequently, discovery claims should be thought of as protoscience.
    Consequently, she succeeded in making a kind of candy only using natural sweeteners.
    Water absorption is greatly reduced; consequently runoff is increased, resulting in accelerated erosion rates.
    Consequently, they are probably more concerned than their predecessors were about job security and economic benefits.
    Consequently—and paradoxically—laws introduced to protect the jobs of ordinary workers may be placing those jobs at risk.
    Consequently, it is likely that a newly developed adhesive will become the routine method of holding most types of cuts closed.
    Instead, I grew up believing that I was supposed to be totally independent and consequently became very reluctant to ask for help.
    We spend a large part of our daily life talking with other people and, consequently, we are very accustomed to the art of conversing.
    And instead, I grew up believing that I was supposed to be totally independent and consequently became very reluctant to ask for help.
    Most fared poorly in school not because they lacked ability but because they found school unchallenging and consequently lost interest.
    Consequently, students living abroad usually have a better and more thorough understanding of the outside world than those who stay at home.
    Consequently, even though the machine is expensive, a dermatological clinic in Westville is considering buying one to reduce diagnostic costs.
    Consequently, much effort has been spent on identifying students with problems in order to give them more attention before they become failures.
    Consequently, the survival rate of lobster larvae must be going up, and the lobster population in Foerkland's coastal waters is bound to increase.
    Confusion consequently reigns over a broad spectrum of unanswered questions dealing with avian origins and the biology of dinosaurs and early birds.
    Consequently, the timing and rhythms of biological functions must closely match periodic events like the solar day, the tides, the lunar cycle, and the seasons.
    Consequently, people who do not talk very easily may be incorrectly understood as being less agreeable than those who have no difficulty keeping up a conversation.
    Consequently, children of a particular age cannot be expected to master educational material without taking into account their current level of cognitive development.
    These days the word "kefir" is consequently more likely to bring to mind glamorous, yoga mat-toting women from Los Angeles than austere visions of blustery Eastern Europe.
    '''

    content_result='''
    They were elated at the result.
    Are you pleased with the result?
    She was delighted at the result.
    The result will be higher taxation.
    I am very pleased about the result.
    The result is entirely unpredictable.
    She died as a result of her injuries.
    The result was a sensational 4–1 victory.
    She could not hide her dismay at the result.
    It was a very creditable result for the team.
    She is waiting on the result of a blood test.
    The result is a humiliation for the president.
    The end result is very good and very successful.
    The result has been a giant leap in productivity.
    They will announce the result of the vote tonight.
    As a result, services have been drastically reduced.
    The final result of the poll will be known tomorrow.
    Their success was the result of years of hard graft.
    This result gives us all the more reason for optimism.
    They were understandably disappointed with the result.
    She predicted (that) the election result would be close.
    Most voters believe the result is a foregone conclusion.
    The result reflects a modest rightward shift in opinion.
    The election result marked the eclipse of the right wing.
    They were bitterly disappointed at the result of the game.
    The violence was the result of political and ethnic conflicts.
    This result is consistent with the findings of Garnett & Tobin.
    Often, such projects are the result of legislative log-rolling.
    Their bodies had suffered contortion as a result of malnutrition.
    Aggression is often the result of fear.
    '''

    content_lead_to='''
    A poor diet will ultimately lead to illness.
    Insults can lead to the incitement of violence.
    He pulled away, extending his lead to 15 seconds.
    It will lead to devaluation of a number of currencies.
    Technological changes will inevitably lead to unemployment.
    Ethnic tensions among the republics could lead to civil war.
    Many factors can lead to growth retardation in unborn babies.
    There were fears that privatization would lead to job losses.
    Excessive pressure can lead to perforation of the stomach wall.
    Policies of tax reduction must lead to reduced public expenditure.
    He agreed that these policies will lead to gridlock in the future.
    Lack of exercise can lead to feelings of depression and exhaustion.
    The group proceeded with a march they knew would lead to bloodshed.
    Salt intake may lead to raised blood pressure in susceptible adults.
    The government cautioned that pay increases could lead to job losses.
    Straining to lift heavy weights can lead to a rise in blood pressure.
    We have learned that a market crash need not lead to economic disaster.
    His visit is expected to lead to the restoration of diplomatic relations.
    They hoped this would lead to the departure of all foreign forces from the country.
    Critics say that the new system may be more economic but will lead to a decline in programme quality.
    It may lead to excessive spending.
    Lead in food may lead to poor health.
    Failure will lead to unhappiness in life.
    Such programmes lead to an associate degree.
    Excessive drinking can lead to stomach disorders.
    Eating too much sugar can lead to health problems.
    New evidence might lead to the conclusion that we are wrong.
    Abuse can lead to both psychological and emotional problems.
    This may eventually lead to routine inoculation of children.
    Food that is insufficiently cooked can lead to food poisoning.
    '''

    content_stem_from='''
    All my problems stem from drink.
    He told the magazine in an exclusive interview: "All my problems stem from drinking."
    Intolerance does not stem from disagreement.
    According to the survey, that lack of enthusiasm could stem from concerns about privacy and security.
    This idea may stem from ghoulish novels.
    May stem from a fetish for animal fur and skins.
    Many of the problems now stem from private banking.
    What opportunities stem from architecture management?
    But this grief does not stem from the child's autism in itself.
    My weak points normally stem from those times when I'm not so organized.
    But if your problems stem from the nature of the work itself, then read on.
    Its flaws, like its strengths, stem from the personalities of its founders.
    However, such disagreements often stem from a simple communication problem.
    Conflicts and food price crises all stem from the degradation of land, he added.
    This difference may stem from improved nutrition or other environmental factors.
    Those fears, it is argued, stem from a double squeeze on America's middle class.
    The optional dimensions stem from the standard keyword categories in the MIML data.
    These four broad areas stem from the sequence of meetings that have been held recently.
    Obvious practical problems in this approach stem from the distributed-memory organization.
    According to Messrs Mann and Ornstein, Congress's current problems stem from three things.
    Part of the problem, he said, may stem from an inherent flaw in the way vitamins are studied.
    I've often wondered: Do these attitudes really stem from innate national or cultural differences?
    Bunnies, eggs, Easter gifts and fluffy, yellow chicks in gardening hats all stem from pagan roots.
    The revisions almost all stem from higher tax revenues, a surge of cash that has shocked everyone.
    Confidence and fear are contradictory states of mind that both stem from our beliefs and attitudes.
    '''

    content_bring='''
    All you need bring are sheets.
    What if she forgets to bring it?
    I didn't know how much to bring.
    Next time I'll bring a book.
    The pictures bring the book alive.
    Bring it into the light so I can see it.
    Bring a coat. It might turn chilly later.
    The scandal may bring down the government.
    Did you bring the contract and (all) that?
    Pressure can bring out the worst in people.
    Flowers can bring a dull room back to life.
    The winter sports bring the jet set to town.
    What did you bring the kids back from Italy?
    He beckoned to the waiter to bring the bill.
    I don't want to bring shame on the family name.
    Members are advised to bring a brown-bag lunch.
    It is inadvisable to bring children on this trip.
    She could not bring herself to tell him the news.
    It'll take months to bring the band up to scratch.
    It may freeze tonight, so bring those plants inside.
    It was decided to bring the meeting forward two weeks.
    They intend to bring their complaints out into the open.
    The reforms will bring benefits, socially and politically.
    Addiction to drugs can bring a multitude of other problems.
    I hope nobody's forgotten to bring their passport with them.
    Everything will be done to bring those responsible to justice.
    Interest-rate cuts have failed to bring about economic recovery.
    How much does she bring in now?
    Old age can bring many problems.
    Waiter, could you bring me some water?
    '''

    content_thanks_to='''
    I would like to propose a vote of thanks to our host.
    It was all a great success—thanks to a lot of hard work.
    Thanks to a new directive, food labelling will be more specific.
    Thanks to Bateman's heroics in the second half, the team won 2–0.
    Thanks to the automobile, Americans soon had a freedom of movement previously unknown.
    I owe a debt of thanks to Joyce Thompson, whose careful and able research was of great help.
    Thanks, thanks to both of you.
    Thanks to mobile libraries, these people can still borrow books.
    In fact, there are often huge "quick win" opportunities, thanks to years of neglect.
    Thanks to the company's increasing third-generation, China Unicom Ltd saw new profits.
    That year, Lincoln's likeness made its debut on the penny, thanks to approval from the U.S.
    Thanks to film, future generations will know the 20th century more intimately than any other period.
    Around the world, people changed sleep patterns thanks to the start or end of daylight savings time.
    Samsung's sales have skyrocketed thanks to a sleek production system that rapidly brings new products to market.
    Thanks to that job I became an avid reader.
    Everyone knows about it now, thanks to you!
    How can I ever express my thanks to you for all you've done?
    She preceded her speech with a vote of thanks to the committee.
    Thanks to the angle at which he stood, he could just see the sunset.
    The delegation was carrying a message of thanks to President Mubarak.
    Thanks to her skilful handling of the affair, the problem was averted.
    It is thanks to this committee that many new sponsors have come forward.
    They were already under stress, thanks to the aftershock of last year's drought.
    Thanks to its Ethernet card, you can link straight into your network at the office.
    Thanks to a group of public-spirited citizens, the Krippendorf garden has been preserved.
    Thanks to fibre optics, it is now possible to illuminate many of the body's remotest organs and darkest orifices.
    Thanks to all of you.
    Thanks to his salty coffee, I'm full now!
    Thanks to Nike, the shoe dream comes true.
    I want to say thanks to our foreign teachers.
    '''

    content_reason='''
    I had no reason to doubt him.
    He had every reason to be angry.
    The reason is blindingly obvious.
    I have good reason to be suspicious.
    There is no reason to disbelieve him.
    Are you angry with me for some reason?
    This was the real reason for her call.
    He said no but he didn't give a reason.
    There seems no reason to doubt her story.
    That's the only reason I'm actually going.
    There is no compelling reason to believe him.
    As always, Peter had a reason for his action.
    I have no particular reason for doubting him.
    They have reason to believe that he is lying.
    I really don't see any reason for changing it.
    He was found not guilty, by reason of insanity.
    There's no earthly reason why you shouldn't go.
    Midge couldn't quite put her finger on the reason.
    They had reason to believe there could be trouble.
    I don't have any particular reason to distrust them.
    For some peculiar reason , she refused to come inside.
    The principal reason for this omission is lack of time.
    Then, for no apparent reason , the train suddenly stopped.
    Their governments have no reason to "massage" the statistics.
    You've no reason to reproach yourself, no reason to feel shame.
    Faith is stronger than reason.
    Why won't you listen to reason ?
    He lost all sense of reason.
    Why can't they see reason ?
    She saw no reason to prevaricate.
    '''

    content_as_a_result_of='''
    She died as a result of her injuries.
    Their bodies had suffered contortion as a result of malnutrition.
    Millions will face starvation next year as a result of the drought.
    As a result of their compatibility, Haig and Fraser were able to bring about wide-ranging reforms.
    As a result of the growing fears about home security, more people are arranging for someone to stay in their home when they're away.
    As a result of it, the state gets many earthquakes.
    The statue fell into the sea when the cliff collapsed as a result of the storm.
    Those that have disappeared were destroyed by fire as a result of lightning or civil war.
    Student's motivation and passion for study has been stimulated as a result of educational reform.
    As a result of this "shrinkage" as the shops call it, the honest public has to pay higher prices.
    She then talks about health issues and the physical damage that can occur as a result of dieting.
    Diabetic neuropathy is damage to the nerves as a result of diabetes, and affects up to 50% of people with diabetes.
    Some mountains were formed as a result of these plates crashing into each other and forcing up the rock at the plate margins.
    Environmentalists' prediction that the world's Merrick population would decline as a result of the spill has proven unfounded.
    None of these volcanoes was formed as a result of collisions between plates of the Martian crust—there is no plate motion on Mars.
    Many scientists believe that El Nino events, caused by warming in the Pacific, are becoming more intense as a result of climate change.
    A small segment of the urban society started to specialize in nonagricultural tasks as a result of the city's role as a regional center.
    Mobile telecommunications possession is expected to double in Shanghai this year as a result of a contract signed between the two companies.
    As a result of divination, Apollo was angry with the Greeks. Because Agamemnon robbed his father's maiden, and his father was the high priest of Apollo.
    In fact, you may have heard about the new "superbugs", which are antibiotic-resistant bacteria that have developed as a result of overprescribed antibiotics.
    As a result of that discrepancy between the privileges they feel they're owed and their inflated sense of self-worth, they don't work as hard for their employer.
    As a result of this and also of the fact that workers' jobs were generally much less secure, distinct differences in life-styles and attitudes came into existence.
    As a result of the above problems, therefore, we feel that the most suitable course of action is to return to you unpaid any of the goods considered unsatisfactory.
    As a result of these diverse methods by which plants supply resources, unique soil communities form under different plant species and under plant communities that differ in composition.
    During the middle of the 19th century, Germany, along with other European nations, experienced an unprecedented rash of workplace deaths and accidents as a result of growing industrialization.
    Many seabirds died as a result of the oil spill.
    2,000 prisoners died as a result of torture and maltreatment.
    Her hair started falling out as a result of radiation treatment.
    Her reputation suffered a mortal blow as a result of the scandal.
    '''

    content_owning_to='''
    Owning to several reasons, the paper also has many deficiencies.
    Owning to the Asian financial crisis the stock prices tumbled in the world.
    This problem would be solved easily if just owning to confining of financial system.
    Owning to their passion, hard work and persistence, they finally succeeded in the contest.
    Owning to his excellence performance in the work, the company decided to raise his salaries and position.
    Nevertheless, owning to the uneven brightness, the image of fuel droplet could not be segmented efficiently.
    At present, we can not undertake to entertain your order owning to the uncertain availability of raw materials.
    At present, we can not undertake to entertain your order owning to the uncertain availability of raw materials.
    Owning to a series of unique advantages, spread spectrum technology has been applied to many corresponding domains.
    However, the application of index is limited owning to lack of the study about formation mechanism in karst cave system.
    At night, the ascent of SAP flow could be observed in the above main tree species owning to the existence of root pressure.
    Owning to history, reality and female student's self aspects, the female university students usually in the inferior position.
    Owning to its superior quality and reasonable price our silk has met with warm reception and quick sale in most European countries.
    In modern society, owning to the emergence of practice negative effect, human beings begins to think practical rationality problems.
    Owning to this effect on fermentation, it can promote the cell growth, at the same time, can accelerate the secondary metabolite biosynthesize.
    Owning to its special properties, trade secret would lose values once being publicized, and the loss caused would be inestimable and irreversible.
    Whereas owning to the various ecological and social environments, the regional characteristics of the proverbs in modem Mongolian are extremely obvious.
    Even though there are abundant Manchu Literatures in Dalian Library, people know little about them owning to imperfect reorganizing work and other reasons alike.
    Owning to the historical limitation, Ansermo is doomed to be a tragic character, the readers can touch not only the failure of "self" but that of the whole society.
    Owning to the predominant advantages in separating azeotropic and close-boiling point mixtures, batch azeotropic distillation is being paid more and more attention.
    It is regrettable to see an order dropped owning to no agreement on price; however, we wish to recommend you another quality at a lower price for your consideration.
    Owning to the limitation of the scope, the data and the studying ability, qualitative analysis is the main method. Whenever necessary, quantitative analysis is utilized.
    Owning to the deficiency of the original experiment method on the teaching material, the rational improvement scheme has been designed, proved the good experiment result.
    The bioconversion of lignocellulosic materials for organic acid had great meaning to both economic and social benefit, owning to its renewable character, abundance and low price.
    Food impaction is the phenomenon appearing in the chewing course when the food dregs or fibers are pushed into the clearance by occlusal force or owning to the gingival shrinkage.
    Time and space are important for relativity study in physics. In matter space exists a gravitational field, owning to which time is varied, space is bent, and light trace is changed.
    '''

    content_due_to='''
    The mare is due to foal today.
    The accident was due to excessive speed.
    They arrived late due to a motorway holdup.
    The coast road is closed due to bad weather.
    Many accidents were due to pilot misjudgment.
    His mistake was due to youth and inexperience.
    She is due to deliver a lecture on genetic engineering.
    The insurance company will refund any amount due to you.
    It was a real prize due to its rarity and good condition.
    Due to official parsimony only the one machine was built.
    He is due to address a conference on human rights next week.
    Grateful thanks are due to the following people for their help...
    Downtime due to worm removal from networks cost close to $450 million.
    Unfortunately, due to unforeseen circumstances, this year's show has been cancelled.
    Filming was due to start next month.
    The trial is due to begin next month.
    Our thanks are due to the whole team.
    The ferry is due to land at 3 o'clock.
    The software is due to ship next month.
    Rose is due to start school in January.
    The results could simply be due to chance.
    The problem may be due to poor workmanship.
    They were due to pay the balance on delivery.
    Most of the problems were due to human error.
    The first plane is due to leave on October 2.
    The breakdown was due to a mechanical failure.
    The problem was due to the simple fact that...
    The factory is due to be demolished next year.
    The team's success was largely due to her efforts.
    Have they been paid the money that is due to them?
    '''

    content_in_consequence_of='''
    "It is largely a consequence of 150 years of industrialization in the major developed countries of the world."
    All I want to stress is that my discovery of her was a fatal consequence of that "princedom by the sea" in my tortured past.
    Humans suffered as a consequence from nutritional deficiencies in almost all parts of the world because there weren't complete diets.
    I want to discuss the consequence of rationality in playing games, slightly philosophical for a few minutes.
    As a consequence, he says little money is available to provide crucial assistance in areas of health, shelter,nutrition,education, water and sanitation and hygiene.
    Without question he's the most terrifying of all of the mourners, and he gives an angry, powerful, vitriolic speech about the terrible state of England - the terrible state of the Church of England and, as a consequence, of the terrible state of England in 1637.
    "In fact,the most direct consequence of releasing them, I believe,would be to further inflame anti-American opinion and to put our troops in greater danger,"
    It stands in a relation of opposition to other regime types and as a consequence the possibility of conflict of tension and war is built in to the very structure of politics.
    But that kind of dynamic in which the choice is just between regimes and oppositions is a consequence of policies that have to be changed.
    As a consequence, the diary measures have their own sets of flaws that we'll talk about in a minute.
    As a consequence, many of these products are ineffective,sub-standard, and in some cases dangerous.
    As a consequence, there's a lot of variability and error in these dietary measurements.
    "We have got an enormous amount of work to do, and people down in the Gulf, particularly businesses, are still suffering as a consequence of this disaster."
    This idea that humans have evolved, that physical changes have occurred in humans as a consequence of their food environment, has led us to where we are now.
    "The unanticipated consequence is that people are now crossing less frequently and have to wait one to two hours coming in and out of our ports of entry."
    As a consequence, he and other researchers then started looking into the lifestyles of people in these various countries to see what might help explain these differences in heart disease.
    They are only a consequence of the president making a judgment that it is in the best interest of the United States of America our national security interests to talk with the Iranian regime,"
    As a consequence, and of course there's stuff on them and in them that changes the calories.
    And I am committed to working with President Calderon to promote the kind of bottom-up economic growth here in Mexico that will allow people , and as a consequence will relieve some of the pressure border,".
    The biodiversity has shrunk for various reasons of profitability and as a consequence of that things like corn crops, and the orange crops in Florida are particularly susceptible to certain environmental conditions, certain pests, and certain types of blight.
    '''

    content_in_view_of='''
    In view of the fact that Hobson was not a trained economist his achievements were remarkable.
    Residents were warned not to be extravagant with water, in view of the low rainfall this year.
    His chances do not seem good in view of the fact that the Chief Prosecutor has already voiced his public disapproval.
    Voluntary work was particularly important in view of the fact that women were often forced to give up paid work on marriage.
    The scientists saw the public as being particularly two-faced about animal welfare in view of the way domestic animals are treated.
    The debate here will be limited in two main respects in view of the time available.
    In view of the technological improvements in the last few years, business will require in the future proportionately fewer workers than ever before.
    In view of the weather, the event will now be held indoors.
    Thanks, but in view of the general responses, you and I are definitely in the minority.
    In view of this behavior it has been suggested that chemicals present in fresh buck rubs may help physiologically induce and synchronize fertility in females that visit these rubs.
    It upheld verdicts in lower courts and ruled that in view of the complex scientific issues involved.
    In view of the present situation, we'll have to revise our original plan.
    And yet, in view of how many things.
    This was understandable in view of my family background.
    In view of our relationship, we would offer you a discount of 5%.
    Eg. In view of our relationship, we would offer you a discount of 5%.
    In view of our long-standing relationship, we agree to allow you a discount.
    Something perhaps, but sadly little in view of the magnitude of the evil.
    So in view of the lukewarm reviews, the acquisition by Microsoft was unexpected.
    Accordingly and in view of the importance of this issue, further discussion is needed.
    Buyers have withdrawn from the market in view of the abrupt turn of the trend of prices.
    In view of the image of God stamped on every person, justice must always be tempered by love.
    In view of our good cooperation over the past few years, we are prepare to accept your price.
    In view of the long business relations between us, we wish to meet you half way to settle the claim.
    In view of the development prospects of the Chinese market at present, he does some market research.
    After some miles they came in view of the clump of trees beyond which the village of Marlott stood.
    It is in view of the foregoing that an account bas been attempted here of the plan for a cottage scale bakery.
    '''

    content_as_a_consequence_of='''
    Maternity services were to be reduced as a consequence of falling birth rates.
    As a consequence of his hard work, he was eventually admitted to Nanjing University.
    Krebs has discovered that great blue herons look up more often when in smaller flocks than when in larger ones, solely as a consequence of poor feeding conditions.
    As a consequence of pathological damage, the liver becomes inflamed.
    This theory views gravity not as a force but as a consequence of the curved geometry of space and time.
    As a consequence of this, the valuation of assets such as bonds and equities should come under pressure.
    This is what seems to happen as a consequence of the so-called "international credit-market crisis."
    Male fetuses were, in other words, being spontaneously aborted-presumably as a consequence of stress.
    General relativity views gravity not as a force but as a consequence of the curved geometry of space and time.
    As a consequence of those assurances, considerable riot control equipment had been loaned to other agencies.
    I know it's hard to fill the hole that occurred as a consequence of you losing folks who you had worked with for so long.
    Suppose that the Update Software Stack scenario will no longer include Step 8 as a consequence of a change in the process.
    Suppose that the Update software stack scenario was not to include Step 8 anymore, as a consequence of a change in the process.
    If the company pays a dividend, a shareholder gets that piece of the company's cash flow as a consequence of share ownership.
    Given that voyage time will increase as a consequence of the reduced speed, the fuel saving will be somewhat less, about 40%.
    For almost a century, the Universe has been known to be expanding as a consequence of the Big Bang about 14 billion years ago.
    History shows today's problems could certainly materialise as a consequence of the failure to provide sufficient economic stimulus.
    The British economy recovered and there was no sign of the surge in inflation that had been predicted as a consequence of devaluation.
    As a consequence of this change in philosophy, pretty much all code that USES Unicode, encodings or binary data most likely has to change.
    The XSRObject is created as a consequence of registering and completing the XML Schema, which is annotated for decomposition in this case.
    Exports of goods and services, the main engine of growth in recent years, will weaken sharply as a consequence of the global downturn.
    VERY low learning curve as a consequence of the previous three points. For the most part, you don't even need to ever read the documentation.
    Sea level rise is expected as a consequence of continuing climate change, which is spurred by human activities including the burning of fossil fuels.
    As a consequence of requirements being refined, risks identified, executable software built, and lessons learned, the set of significant elements may change.
    It also supports complex event processing in which a series of events may be observed to produce one event as a consequence of the relationships in the series.
    '''

    content_on_account_of='''
    The president declined to deliver the speech himself, on account of a sore throat.
    He felt guilty on account of his former anger at the kind gentleman.
    The fact was, he could not refuse the request, on account of the dwarf's third gift.
    Sorry that I cannot go to the airport to pick you up in person on account of stomach ache.
    If he had only known then the dreadful things that were to happen to him on account of his disobedience!
    Heidi had already great confidence in the butler, especially on account of the resemblance she had discovered.
    The Apollo 9 Command module was called Gumdrop, on account of the blue cellophane wrapping in which the craft was delivered.
    She retired early on account of ill health.
    It was all on account of the whiskey and the excitement, I reckon.
    The flight was postponed on account of bad weather.
    Chettiar was released on account of illness.
    All on account of a ridiculous error.
    This is all on account of climate change.
    The manager had to step down on account of poor health.
    He explained he made a mistake on account of his illness.
    Did you choose this company on account of high pay (remuneration)?
    Peter replied, "Even if all fall away on account of you, I never will."
    She petitioned for divorce on account of the breakdown of their marriage.
    ECOWAS suspended Niger in October on account of Mr Tandja's bad behaviour.
    So far I haven't heard of anybody who wants to stop living on account of the cost.
    On account of me you will stand before governors and Kings as witnesses to them.
    Likewise the proposed Motor Pool Improvement, on account of a shard of Colonial crockery.
    In the cloudless night ground temperature decreases on account of the heat lost to the space.
    The child could not be seen on account of her small size, but the head of her doll was visible.
    He took all afternoon on each one of them, on account of having to hunt for each letter on the typewriter.
    The Beckhams are invited on account of David's work with Prince William on England's 2018 World Cup bid.
    That was largely on account of more women joining the labor force, and continuing to work once they had kids.
    Chinese fruits are very popular in the market on account of their superior quality and competitive price.
    '''

    content_that_s_why='''
    That's why I left so early.
    That's why I must trust you to keep this secret.
    That's why most people and organizations just roll over and give up when they're challenged or attacked by the I.R.S.
    That's why tourists come here.
    That's why I chose this lodge.
    That's why it is called a UFO!
    That's why the glass breaks.
    That's why we call it black tea.
    That's why I draw cartoons.
    That's why we have plastic surgery!
    That's why he is called the emperor.
    That's why we chose him for the job.
    That's why he hates it.
    That's why we have a clock in system.
    That's why you will feel uncomfortable.
    That's why you never tire of traveling.
    That's why school reform is so critical.
    That's why he works in a comic book store.
    That's why most teenagers are short sighted.
    That's why I transferred to this university.
    I guess that's why I forgot all about the time.
    Do you think that's why he's been absent recently?
    That's why automation is key to precision farming.
    Maybe that's why many Russian children eat soup for lunch.
    "It's a Cheshire cat," said the Duchess, "and that's why."
    That's why many sensation-seekers also like extreme sports.
    That's why it got high scores in physics, history and math.
    That's why I want to focus on marketing in graduate school.
    Everybody makes mistakes; that's why they put erasers on pencils.
    '''

    content_so_that='''
    He angled his chair so that he could sit and watch her.
    Mari tilted her head back so that she could look at him.
    They're dredging the harbour so that larger ships can use it.
    She burned the letters so that her husband would never read them.
    The boys dug pits and baited them so that they could spear their prey.
    I studied Italian so that I would be able to read Dante in the original.
    We perfected a hand-signal system so that he could keep me informed of hazards.
    Bend the wire so that it forms a 'V'.
    Can you work it so that we get free tickets?
    Bess spoke up, smiling so that her dimples showed.
    Swaddle your newborn baby so that she feels secure.
    I bored a hole so that the bolt would pass through.
    They brought her meat so that she never went hungry.
    I'll give you a key so that you can let yourself in.
    Pack your suit carefully so that you don't crease it.
    Stir the sauce constantly so that it does not separate.
    Fasten the gates securely so that they do not blow open.
    I stood the little girl on a chair so that she could see.
    I moved my legs out of the way so that she could get past.
    She worked hard so that everything would be ready in time.
    They think the attacker was very tall—so that lets you out.
    He spun the steering-wheel so that we yawed from side to side.
    We should be able to wangle it so that you can start tomorrow.
    All the materials are on site so that work can start immediately.
    We're trying to swing it so that we can travel on the same flight.
    Your task is to interchange words so that the sentence makes sense.
    They wrote the notice in big red letters so that it would stick out.
    Defrost the fridge regularly so that it works at maximum efficiency.
    The whole idea of going was so that we could meet her new boyfriend.
    Her parents made sacrifices so that she could have a good education.
    '''

    content_in_this_way='''
    Many linguists have looked at language in this way.
    Each year, hundreds of animals are killed in this way.
    I couldn't believe these people were behaving in this way.
    Only in this way can the critical mass of participation be reached.
    Considerable efforts have been made to rehabilitate patients who have suffered in this way.
    In this way, it learned to paint.
    In this way, they provided credit.
    In this way the body cures itself.
    In this way, kids can learn by themselves.
    Only in this way can the experiment succeed.
    In this way, they grew stronger and stronger.
    Is it dangerous to move around the world in this way?
    He pulled my sleeve, attracted my attention in this way.
    In this way, you will have chances to be close to nature.
    In this way, they broke off pieces from one of the stones.
    In this way, you don`t have to review all the material again.
    In this way, you actually have a plan you can remember easily.
    In this way, his students learned more than seven hundred words.
    Seals do not click in this way, but they do have remarkable whiskers.
    In this way, they are bound to lead more meaningful and successful lives.
    He reads the texts every morning. In this way, he is able to recite them.
    Only a small fraction of the creatures that die are preserved in this way.
    In this way, you will be able to meet new friends of different age groups.
    Now, there are some very good reasons to approach the material in this way.
    Only in this way, can they have a fruitful and successful study life abroad.
    In this way, you can increase the number of words and know the dialogues better.
    Chinese doctors believe that they can control the body's natural forces in this way.
    They share their notes to help students prepare for High School Entrance Exams in this way.
    To be clear, people can't keep away from all the risks in this way, but it's the best choice.
    Disney now has ten "franchises" that it treats in this way, from Mickey Mouse to Disney Fairies.
    '''

    content_in_that='''
    She looks a sight in that hat!
    She won't last long in that job.
    You look so pretty in that dress!
    She's the peacekeeper in that family.
    We could pitch our tent in that field.
    I'm lucky in that I've got four sisters.
    Don't work in that department; it's a madhouse.
    At least one hostage was beheaded in that room.
    They've lived in that house for nigh on 30 years.
    It bothers me to think of her alone in that big house.
    We had some wine—or what passes for wine in that area.
    I've heard tales of people seeing ghosts in that house.
    That's better. I was suffocating in that cell of a room.
    We got really cruddy service in that restaurant last time.
    The dead and wounded in that one attack amounted to 6 000.
    The creaking of the hinges sounded very loud in that silence.
    There are a lot of small businesses starting up in that area.
    We've made extraordinary progress as a society in that regard.
    Clean water is a precious commodity in that part of the world.
    We were all living on top of each other in that tiny apartment.
    There were some holes in that theory, some unanswered questions.
    "God knows," she muttered, "what's happening in that madman's mind."
    A large proportion of the dolphins in that area will eventually die.
    You look very smart in that suit.
    You look sensational in that dress!
    You must be boiling in that sweater!
    Where are you going in that rig-out?
    Careful you don't tread in that puddle.
    It's in that drawer.
    There's more than a grain of truth in that.
    '''

    content_on_the_grounds_that='''
    He declined further comment on the grounds that the case was sub judice.
    The case was dismissed on the grounds that there was not enough evidence.
    The board has banned the film on the grounds that it contravenes criminal libel laws.
    The court overturned that decision on the grounds that the prosecution had withheld crucial evidence.
    Other archaeologists have criticized those conclusions on the grounds that passenger pigeon bones would not be likely to be preserved.
    Despite these finding, we are urged to support monopolistic power on the grounds that such power creates an environment supportive of innovation.
    This law was declared unconstitutional by the Supreme Court on the grounds that general taxes were being collected to pay one special group of people.
    Successive governments have permitted such increases on the grounds that the cost of investing in and running the rail network should be borne by those who use it, rather than the general taxpayer.
    In an advertisement for Presorbin, its makers argue that Presorbin is superior on the grounds that doctors have written 200 million prescriptions for Presorbin, as compared to 100 million for Veltrex.
    New York judge denied him bail on the grounds that he posed a flight risk.
    A New York judge denied him bail on the grounds that he posed a flight risk.
    But the prosecutor declined on the grounds that the issue should be investigated in the US.
    All the others refused on the grounds that they had to respect the privacy of their donors.
    But Origin's board has already rejected the bid, on the grounds that it undervalues the firm.
    Choose not to expose the service on the grounds that it cannot meet business service level requirements.
    Others dismiss the worry on the grounds that creating new forms of life is not the same as creating life.
    Arizona never adopted the new time plan on the grounds that it was too hot out to "spring forward" an hour in the summer.
    Or it might dismiss the complaint on the grounds that the plaintiffs did not have the right to lodge it in the first place.
    She liked what she saw and booked the studio plus a three-bed cottage on the grounds that had been converted from a cowshed.
    Handelsbanken eschews bonuses too, on the grounds that they work against long-term relationships with customers and employees.
    The court in 1993 endorsed the Maastricht treaty on the grounds that a single currency did not turn the EU into a federal state.
    If a woman is denied treatment on the grounds that she could not possibly benefit, the basis of that denial had better be rock-solid.
    Justice Hugh Bennett said the petition for divorce by Mills was granted on the grounds that the couple had lived apart for two years.
    Phoebe, an avowed champion of animal rights, was persuaded on the grounds that she’d be saving a dog that might otherwise be put down.
    They arrested him and dragged him away to a local clerical court, on the grounds that his sprouting hairdo was a dangerous Western import.
    But there are also conservatives who reject this myth of the Fall on the grounds that every age has been just as dreadful as every other.
    '''

    content_as_long_as='''
    We'll go as long as the weather is good.
    There's a list of repairs as long as your arm.
    As long as you're happy, that's all that matters.
    As long as you have your health, nothing else matters.
    What I really want is to live healthily for as long as possible.
    Tiles can be fixed to any surface as long as it's flat, firm and dry.
    Anyone can be an astrologer as long as they are mathematically minded.
    You are welcome to come and stay as long as you give us plenty of notice.
    Vijay Singh says the course will not play as long as the yardage indicates.
    He said he would still support them, as long as they didn't break the rules.
    I told Norman I would invest in his venture as long as he agreed to one proviso.
    As long as mist hangs o'er the mountains, the deeds of the brave will be remembered.
    As long as staff members are well-groomed, it does not matter how long their hair is.
    Nothing will change as long as the workers continue to accept these appalling conditions.
    As long as the operative word is "greed," you can't count on people keeping the costs down.
    It was as long as the blackboard.
    I'm fine as long as you are happy.
    I reckon we're safe as long as we keep mum.
    I shall never learn it here as long as I live.
    As long as I live, I have enough for me and the child.
    An apartment, probably, as long as it has two bedrooms.
    He will never be good for anything as long as he lives!
    Stay as long as you like.
    You may stay as long as you please.
    We will fight for as long as it takes.
    I've known her at least as long as you have.
    They're welcome to stay here as long as they like.
    Customers often defer payment for as long as possible.
    We can stay here for as long as our supplies hold out.
    '''

    content_in_order_to='''
    She may have farmed the child out in order to remarry.
    My friends endured tremendous danger in order to help me.
    In order to take that job, you must have left another job.
    I'd taken my phone off the hook in order to get some sleep.
    They spread disinformation in order to discredit politicians.
    They were forced to practise cannibalism in order to survive.
    She became a prostitute in order to pay for her cocaine habit.
    Clearly, the police cannot break the law in order to enforce it.
    He was going to college at night, in order to become an accountant.
    In order to maximize profit, the firm would seek to maximize output.
    They're putting up new hotels in order to boost tourism in the area.
    Prices have risen in order to offset the increased cost of materials.
    He lowballed the cost of the project in order to obtain federal funding.
    It's embarrassing the extremes he'll go to in order to impress his boss.
    In order to survive the competition a company should be proactive not reactive.
    They are breaking the law in order to obtain an advantage over their competitors.
    The next hearings will be structured differently in order to minimize the inquisitorial atmosphere.
    Is it in order to speak now?
    We needed the break in order to recharge.
    We stood up in order to get a better view.
    I go swimming every day in order to keep fit.
    She arrived early in order to get a good seat.
    Teamwork is required in order to achieve these aims.
    I had to dismantle the engine in order to repair it.
    The organisms were forced to adapt in order to survive.
    He only gave his consent in order to gratify her wishes.
    He arranged the accident in order to fake his own death.
    In order to be successful he would have to exert himself.
    She wanted to take a career break in order to have children.
    '''

    content_for_the_purpose='''
    I find the best tool for the purpose is a pair of shears.
    Do you come to China for the purpose of studying or traveling?
    He wants to emphasize Hanson himself has a farm that specializes in grapes for the purpose of producing raisins.
    A meeting was called for the purpose of appointing a new treasurer.
    It is one of the most difficult fields to contain for the purpose of statistical quantification.
    The water is brought to the surface, as a liquid or steam, through holes drilled for the purpose.
    The flat character is introduced solely for the purpose of allowing the round character to show off.
    The greatest problem in using intelligence tests for the purpose of prediction is that no dependable criterion of their accuracy exists.
    Article 2 For the purpose of the regulations.
    Listing 7 is example client code for the purpose.
    For the purpose of testing, we'll keep this value at 15.
    Creating accounts for the purpose of selling those accounts.
    You can use either query for the purpose of this example.
    For the purpose of this article, we have simplified the code.
    For the purpose of this exercise, the default workspace is acceptable.
    See previous articles in this series for the purpose of emulators.
    For the purpose of this article, you'll build your application by hand.
    The problem is that vessels are often used not for the purpose intended.
    The trainers conducted weekly classes in the center built for the purpose.
    They must retrench their expenditure for the purpose of making up the deficit.
    We are now writing you for the purpose of establishing business relations with you.
    Layouts featured in this post were created specifically for the purpose of this article.
    For the purpose of this article, only nanosecond calculations will be described in detail.
    For the purpose of this column, however, I'll stick to importing the participant list.
    Let me call it, for the purpose here, subordination, or if you wish, the executive function.
    Imagine if you were tasked to modify this method for the purpose of adding a new feature.
    However, for the purpose of our example it is useful to illustrate all the notifications emitted.
    It can hold real data from a database, but for the purpose of simplicity, a static list is used.
    This routine begins an inquiry context for the purpose of returning group members to the caller.
    '''
    with open('111.txt', 'r', encoding='utf-8') as f:
        # open为打开文件，r为读取
        f = open('111.txt', 'r', encoding='utf-8')
        # 逐行读取文件内容
        lines = f.readlines()
    extractor = CausalityExractor()
    datas = extractor.extract_main(lines)
    for data in datas:
        print('******'*4)
        print('cause', ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]]))
        print('tag', data['tag'])
        print('effect', ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]]))

test()
