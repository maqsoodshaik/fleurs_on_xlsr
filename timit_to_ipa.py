from ipapy import is_valid_ipa
timit_2_ipa = {"aa":"ɑ",
"ae":"æ",
"ah":"ʌ",
"ao":"ɔ",
"aw":"aʊ",
"ax":"ə",
"axr":"ɚ",
"ay":"aɪ",
"eh":"ɛ",
"er":"ɝ",
"ey":"eɪ",
"ih":"ɪ",
"ix":"ɨ",
"iy":"i",
"ow":"oʊ",
"oy":"ɔɪ",
"uh":"ʊ",
"uw":"u",
"ux":"ʉ",
"b":"b",
"ch":"ʧ",
"d":"d",
"dh":"ð",
"dx":"ɾ",
"el":"l̩",
"em":"m̩",
"en":"n̩",
"f":"f",
"g":"g",
"hh":"h",
"jh":"ʤ",
"k":"k",
"l":"l",
"m":"m",
"n":"n",
"ng":"ŋ",
"nx":"ɾ̃",
"p":"p",
"q":"ʔ",
"r":"ɹ",
"s":"s",
"sh":"ʃ",
"t":"t",
"th":"θ",
"v":"v",
"w":"w",
"wh":"ʍ",
"y":"j",
"z":"z",
"zh":"ʒ",
"ax-h":"Dv[ə]([ə̥])",#Devoiced [ə] ([ə̥])
"eng":"Sy[ŋ]",#Syllabic [ŋ]
"hv":"V[h]",#Voiced [h]
"bcl":"[b]cl",#[b] closure
"dcl":"[d]cl",
"gcl":"[g]cl",
"kcl":"[k]cl",
"pcl":"[p]cl",
"tcl":"[t]cl",
"pau":"Pau",#Pause
"epi":"Epi",#Epenthetic silence
"h#":"marker"}#Begin/end marker
# for k in timit_2_ipa.values():
#     print(f'{k}:{is_valid_ipa(k)}')
# print(len(timit_2_ipa))