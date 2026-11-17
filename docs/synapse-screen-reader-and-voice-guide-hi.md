# Synapse: स्क्रीन रीडर और वॉयस कंट्रोल की विस्तृत यूज़र गाइड

अपडेट: 16 नवंबर 2026। यह गाइड सक्रिय `synapse-web` ऐप के मौजूदा कोड पर आधारित है। इसमें Windows और NVDA के साथ कीबोर्ड से ऐप चलाने, बोलकर कमांड देने और दोनों तरीकों को साथ इस्तेमाल करने के निर्देश हैं। लाइव माइक्रोफ़ोन और स्क्रीन रीडर पर हर चरण का परीक्षण नहीं किया गया है।

गाइड हिंदी में है। ऐप के बटन और बोले जाने वाले कमांड अंग्रेज़ी में दिए हैं, ताकि आप उन्हें ऐप में पहचान सकें। अभी वॉयस कमांड **अंग्रेज़ी में बोलें**। उदाहरण किसी एक तय वाक्य तक सीमित नहीं हैं; उनका अर्थ और मौजूदा स्क्रीन दोनों मायने रखते हैं।

## 1. पहले इन तीन चीज़ों का अंतर समझें

- **स्क्रीन रीडर:** NVDA पेज का टेक्स्ट, बटन का नाम, चयन और फ़ोकस पढ़ता है। इससे आप बिना माइक्रोफ़ोन के भी ऐप चला सकते हैं।
- **वॉयस कंट्रोल:** आप माइक्रोफ़ोन में “Play the lecture” जैसी कमांड बोलते हैं और Synapse कार्रवाई करता है।
- **वॉयस नोट:** आप पढ़ाई से जुड़ी अपनी बात रिकॉर्ड करके सहेजते हैं। रिकॉर्ड की गई बात ऐप की कमांड नहीं बनती।

लेक्चर में शिक्षक की आवाज़, Synapse की explanation और आपकी रिकॉर्डिंग अलग ऑडियो हैं। कोई explanation सुनने से पहले सही लेक्चर और सही moment चुना होना चाहिए। Moment का मतलब लेक्चर का वह समय है जहाँ कोई चित्र, समीकरण, ग्राफ़ या दूसरी महत्वपूर्ण जानकारी मिली है।

## 2. इस्तेमाल से पहले तैयारी

1. Synapse का चालू ऐप खोलें। लोकल सेटअप में सामान्य पता `http://localhost:5173` है; अपने चल रहे ऐप का वास्तविक पता इस्तेमाल करें।
2. स्क्रीन रीडर से चलाने के लिए NVDA शुरू करें। अंग्रेज़ी UI पढ़ने वाली आवाज़ उपलब्ध रखें।
3. वॉयस कंट्रोल और नोट रिकॉर्डिंग के लिए माइक्रोफ़ोन जोड़ें। पहली बार अनुमति माँगे जाने पर सही साइट या ऐप के लिए Allow चुनें।
4. हेडफ़ोन लगाएँ, ताकि शिक्षक, Synapse या NVDA की आवाज़ माइक्रोफ़ोन में वापस न जाए।
5. वॉयस कमांड देने से पहले NVDA की जारी आवाज़ रोकने के लिए Ctrl दबाएँ। इससे Synapse या शिक्षक की आवाज़ नहीं रुकती।
6. सर्वर उपलब्ध होना चाहिए। ऐप खुलना भर इस बात की गारंटी नहीं है कि खोज, explanation, वॉयस पहचान या quiz तैयार है।

यदि आपको ऐप शुरू करना है, तो डेवलपर के लिए [स्टार्टअप और मैनुअल जाँच के निर्देश](synapse-web-manual-test-guide.md#1-start-the-correct-app) उपलब्ध हैं। सामान्य उपयोग में ऐप खुला होने पर नीचे के चरणों से शुरू करें।

## 3. स्क्रीन रीडर से चलाने की बुनियाद

### 3.1 पेज पर घूमना

NVDA की सामान्य Windows सेटिंग में ये कुंजियाँ उपयोगी हैं। NVDA कुंजी आम तौर पर Insert या, कॉन्फ़िगरेशन के अनुसार, Caps Lock होती है।

| कुंजी | काम |
| --- | --- |
| Tab | अगले उपलब्ध कंट्रोल पर फ़ोकस |
| Shift+Tab | पिछले कंट्रोल पर फ़ोकस |
| Enter | फ़ोकस किए बटन या लिंक को सक्रिय करना |
| H / Shift+H | Browse mode में अगला / पिछला heading |
| D / Shift+D | Browse mode में अगला / पिछला landmark |
| B / Shift+B | Browse mode में अगला / पिछला button |
| F / Shift+F | Browse mode में अगला / पिछला form field |
| NVDA+F7 | Elements list खोलना |
| NVDA+Space | Browse mode और Focus mode बदलना |
| Ctrl | NVDA की मौजूदा speech रोकना |
| Arrow keys | टेक्स्ट पढ़ना या फ़ोकस किए radio/tab जैसे कंट्रोल का चयन बदलना |

ऐप खुलने पर **Skip to main content** लिंक मिले तो Enter दबाकर मुख्य सामग्री पर जाएँ। Heading पढ़कर पता लगाएँ कि आप Home, Lecture, Timeline, Library या किसी दूसरी स्क्रीन पर हैं।

**Space के बारे में विशेष ध्यान:** Synapse में Talk उपलब्ध होने पर टेक्स्ट फ़ील्ड के बाहर Space वॉयस कंट्रोल शुरू या बंद कर सकता है, भले ही फ़ोकस किसी बटन पर हो। बटन और लिंक चलाने के लिए Enter इस्तेमाल करें। NVDA Browse mode कभी Space को स्वयं संभाल सकता है; इसलिए स्क्रीन रीडर के साथ वॉयस शुरू करने का स्पष्ट तरीका Alt+T है। Radio विकल्प चुनने के लिए Arrow keys इस्तेमाल करें।

### 3.2 ऐप के शॉर्टकट

| शॉर्टकट | काम | कहाँ इस्तेमाल करें |
| --- | --- | --- |
| Alt+T | वॉयस कमांड शुरू / बंद | जहाँ Talk उपलब्ध है; टेक्स्ट फ़ील्ड में भी |
| Alt+S | लेक्चर लिंक / खोज फ़ील्ड पर फ़ोकस | Home और subject results |
| Alt+P | शिक्षक का लेक्चर play / pause / resume | Lecture |
| Alt+L | Explanation audio सुनना / pause करना | Lecture |
| Alt+N | अगला moment | Lecture |
| Alt+B | पिछला moment | Lecture |
| Alt+R | Explanation के टेक्स्ट की शुरुआत पर फ़ोकस | Lecture |
| Alt+. | Explanation audio की गति बढ़ाना | Lecture |
| Alt+, | Explanation audio की गति कम करना | Lecture |
| Alt+M | अपना वॉयस नोट शुरू / बंद करना | Lecture |
| Alt+K | Lecture shortcuts की सूची सुनना | Lecture |

Alt+B bookmark नहीं करता। Alt+R ऑडियो replay नहीं करता; यह स्क्रीन रीडर से टेक्स्ट पढ़ने के लिए है। Lecture shortcuts आम तौर पर टेक्स्ट बॉक्स में टाइप करते समय लागू नहीं होते। Help खुलने पर पीछे की स्क्रीन के शॉर्टकट बंद रहते हैं।

शॉर्टकट काम न करे तो पहले सक्रिय स्क्रीन और फ़ोकस जाँचें। फिर Tab से उसी काम का बटन खोजकर Enter दबाएँ। इससे पढ़ाई आगे जारी रख सकते हैं।

## 4. वॉयस कंट्रोल कैसे इस्तेमाल करें

### 4.1 एक कमांड देने का तरीका

1. NVDA बोल रहा हो तो Ctrl से उसकी आवाज़ रोकें।
2. Alt+T दबाएँ और छोड़ दें। उसे दबाए न रखें।
3. सुनने की tone आने दें। Getting ready के दौरान बोलना शुरू न करें।
4. अंग्रेज़ी में एक साफ़ कमांड बोलें, जैसे “Open my library.”
5. बोलकर रुक जाएँ या Alt+T दोबारा दबाकर रिकॉर्डिंग समाप्त करें।
6. Working के बाद ऐप की प्रतिक्रिया सुनें और परिणाम जाँचें।

वैकल्पिक रूप से, टेक्स्ट फ़ील्ड के बाहर Space एक बार दबाकर बोलना शुरू और दोबारा दबाकर बंद कर सकते हैं। Space दबाए रखकर बोलना और छोड़कर समाप्त करना भी समर्थित है। Talk बटन पर Enter भी start/stop करता है। स्क्रीन रीडर के साथ Alt+T प्राथमिक तरीका रखें।

बटन की स्थितियाँ आम तौर पर इस क्रम में बदलती हैं:

`Talk → Getting ready… → Listening… → Working… → Talk`

कुछ न बोलने पर लगभग 6 सेकंड में turn बंद हो सकता है। एक utterance की अधिकतम अवधि लगभग 12 सेकंड है। बोलने के बाद छोटी चुप्पी भी रिकॉर्डिंग पूरी कर सकती है। लंबे सवाल को छोटा और स्पष्ट रखें।

### 4.2 सवाल के बाद अपने आप सुनना

मौजूदा ऐप में auto-listen सामान्यतः चालू है। Synapse कोई clarification, confirmation या explanation का सवाल बोलने के बाद उत्तर के लिए माइक्रोफ़ोन अपने आप खोल सकता है। **Tone और Listening स्थिति आने पर ही जवाब दें।**

यह लगातार सुनने वाला wake-word सिस्टम नहीं है। यदि अपने आप Listening न आए, तो Alt+T से नया turn शुरू करें। Screen-reader-only feedback preference होने पर auto-listen बंद रहता है, ताकि माइक्रोफ़ोन स्क्रीन रीडर को रिकॉर्ड न करे। Settings में इस preference को बदलने का सामान्य selector अभी नहीं है।

### 4.3 रोज़ काम आने वाली कमांड

| बोलें | क्या होगा |
| --- | --- |
| “Where am I?” | मौजूदा स्क्रीन की जानकारी |
| “Help” | इस स्क्रीन की कमांड का छोटा spoken guide |
| “Say that again” | पिछला assistant response दोहराना |
| “Stop talking” | Assistant की speech रोकना |
| “Cancel” | Pending वॉयस अनुरोध / समर्थित confirmation रद्द करना |
| “Go to the home screen” | Home खोलना |
| “Open my library” | Library खोलना |
| “Open bookmarks” | Bookmarks खोलना |
| “Open my voice notes” | Voice Notes खोलना |

“Stop talking” NVDA को बंद नहीं करता। “Cancel” किसी processing job को रद्द करने का सार्वभौमिक कमांड नहीं है। काम तभी सफल मानें जब सही स्क्रीन, चयन या वास्तविक playback भी बदल गया हो।

## 5. पहली बार: अकाउंट और Welcome

### स्क्रीन रीडर और कीबोर्ड से

1. Sign in या Create account स्क्रीन का heading पढ़ें।
2. Tab से Username और Password फ़ील्ड पर जाएँ और जानकारी भरें।
3. Username 3–64 characters का होना चाहिए। अक्षर या अंक से शुरू करें; dot, dash और underscore भी स्वीकार हैं।
4. Password 8–128 characters का रखें। जरूरत हो तो Show password कंट्रोल इस्तेमाल करें।
5. सही submit बटन पर Enter दबाएँ। त्रुटि आए तो संबंधित फ़ील्ड और संदेश पढ़कर ठीक करें।
6. Welcome पर Accessibility mode या Study mode चुनें और Get started / Continue दबाएँ। Skip for now मिले तो उसे भी चुन सकते हैं।

यह local account है। इसे email login न समझें; अभी password recovery का सामान्य flow उपलब्ध नहीं है।

### वॉयस से उपलब्ध सहायता

- “Create an account” से signup स्क्रीन खोल सकते हैं।
- “I already have an account” से sign-in स्क्रीन पर जा सकते हैं।
- “My username is maha22” जैसे वाक्य से username भर सकते हैं। ऐप के दोहराए हुए अक्षर सुनकर spelling जाँचें। गलत हो तो फ़ील्ड में सुधारें।
- **Password कीबोर्ड से टाइप करें**, फिर Enter या submit बटन इस्तेमाल करें। ऐप password वॉयस से नहीं लेता।
- Welcome पर “Accessibility mode” या “Study mode”, फिर “Get started” बोल सकते हैं। “Skip the introduction” भी इस्तेमाल कर सकते हैं।

Accessibility mode में वीडियो छिपा रहता है, लेकिन शिक्षक का ऑडियो चलता है। Study mode में वीडियो दिखता है। लेक्चर स्क्रीन से mode बाद में बदल सकते हैं।

## 6. लेक्चर खोजना और खोलना

### 6.1 स्क्रीन रीडर से

1. Home पर जाएँ और Alt+S दबाएँ।
2. विषय लिखें, जैसे `electric fields`, या YouTube लेक्चर का लिंक paste करें।
3. संबंधित Search / Start learning कंट्रोल सक्रिय करें।
4. खोज के परिणाम आएँ तो शीर्षक और उपलब्ध channel जानकारी पढ़ें। सही result पर Enter दबाएँ। Result का visible open control processing सीधे शुरू कर सकता है।
5. Processing status पढ़ें। Explanation तैयार होने और लेक्चर खुलने में अलग-अलग समय लग सकता है।
6. Retry, reconnect या cancellation जैसे कंट्रोल दिखाई दें तो उनकी स्थिति पढ़कर इस्तेमाल करें। Cancel processing के लिए उपलब्ध बटन इस्तेमाल करें।

### 6.2 वॉयस से

1. “Go to the home screen.”
2. “Search for lectures about electric fields.”
3. परिणाम आने के बाद “Read the results.” सूची लंबी हो तो “Read more.”
4. “Choose the first result.”
5. **इसके बाद “Start learning” बोलें।** वॉयस से result चुनना केवल उसका URL चुन सकता है; उससे लेक्चर शुरू हो जाना न मानें।
6. लेक्चर स्क्रीन खुलने पर “Play the lecture.”

पहले से कॉपी किया YouTube लिंक हो तो “Paste the link”, फिर “Start learning” बोलें। Clipboard अनुमति न मिले तो Alt+S, Ctrl+V और संबंधित बटन इस्तेमाल करें। Home पर subject browsing के लिए “Show me physics lectures” कह सकते हैं। पुराना उपलब्ध लेक्चर जारी रखने के लिए “Continue where I left off” कहें।

Result number उसी मौजूदा सूची का होता है। नई खोज या पेज बदलने के बाद पुराने नंबर से उसी लेक्चर के खुलने की उम्मीद न करें।

## 7. लेक्चर और explanation को संभालना

### 7.1 शिक्षक का ऑडियो

| कीबोर्ड / कंट्रोल | वॉयस उदाहरण | काम |
| --- | --- | --- |
| Alt+P या Play lecture | “Play the lecture” | शिक्षक का लेक्चर चलाना |
| Alt+P या Pause lecture | “Pause the lecture” | शिक्षक का लेक्चर रोकना |
| उपलब्ध seek controls | “Skip forward thirty seconds” | लेक्चर आगे करना |
| उपलब्ध seek controls | “Go back ten seconds” | लेक्चर पीछे करना |
| Alt+N | “Go to the next moment” | अगले explanation moment पर जाना |
| Alt+B | “Go back to the previous moment” | पिछले moment पर जाना |

वॉयस capture शुरू होने से पहले ऐप अपने मीडिया को pause करने की कोशिश करता है। साधारण कमांड के बाद पहले चल रहा मीडिया वापस चल सकता है। Explicit pause या navigation के बाद paused रहना उचित है। वास्तव में क्या चल रहा है, उसकी आवाज़ और स्थिति जाँचें।

### 7.2 Explanation के सवालों का जवाब

महत्वपूर्ण visual पर लेक्चर रुककर सवाल पूछ सकता है। पूरा सवाल सुनें। केवल उस समय दिए विकल्पों में से चुनें। नीचे के वॉयस उत्तर tone के बाद दें; Listening न आए तो Alt+T दबाएँ। स्क्रीन रीडर से वही नाम वाले answer buttons पर Enter भी दबा सकते हैं।

| ऐप की स्थिति | उत्तर | परिणाम |
| --- | --- | --- |
| Explanation तैयार है | “Listen” | Explanation audio सुनना |
| Explanation तैयार है | “Skip” | इसे छोड़कर लेक्चर जारी रखना |
| Explanation अभी तैयार हो रही है | “Wait” | लेक्चर paused रखकर इंतज़ार |
| Explanation अभी तैयार हो रही है | “Resume” | तैयारी के दौरान लेक्चर जारी रखना |
| जारी लेक्चर के दौरान तैयार होने की सूचना | “Listen” / “Later” | अभी सुनना / बाद के लिए छोड़ना |
| Explanation समाप्त; पूछा गया कि स्पष्ट थी? | “Yes” | लेक्चर या शेष review आगे बढ़ाना |
| वही सवाल | “No” | दूसरी depth के विकल्प लेना |
| वही सवाल | “Replay” | Explanation दोबारा सुनना |
| दूसरी depth चुनने का सवाल | “Brief”, “Standard” या “Detailed” | दिए विकल्प में से version माँगना |
| तैयारी असफल | “Retry”, जब पेश हो / “Resume” | फिर प्रयास / लेक्चर जारी |
| Saved content में version उपलब्ध नहीं | “Process” / “Later” | Missing version बनाना / saved content के साथ रहना |
| Audio विफल | “Listen”, जब पेश हो / “Resume” | फिर audio की कोशिश / आगे बढ़ना |
| लेक्चर के अंत में बाकी explanations का review | “Next” / “Finish”, जब पेश हों | अगला review item / review समाप्त |

“Yes” हमेशा play का पर्याय नहीं है। इसका अर्थ सक्रिय सवाल या confirmation से तय होता है। Explanation पूरी होने के बाद लेक्चर अपने आप चल जाएगा, ऐसा न मानें; स्पष्ट उत्तर दें।

### 7.3 खुद explanation माँगना और पढ़ना

- “Please explain this part” या “I do not understand this, explain it” से मौजूदा संदर्भ की explanation माँगें। Depth न बताएँ तो Standard इस्तेमाल होती है।
- “Give me a brief explanation” या “Give me the detailed version” से दूसरी depth माँगें।
- “Read the explanation to me” या “Listen to the explanation” **ऑडियो चलाता है**।
- “Pause the explanation” से explanation audio रोकें।
- “Read that explanation again” से शुरुआत से replay करें।
- “Speak faster” / “Speak slower” explanation की गति बदलते हैं। ये शिक्षक के वीडियो की गति के कमांड नहीं हैं।

**NVDA से चुपचाप टेक्स्ट पढ़ने का तरीका:** शिक्षक और explanation audio रोकें, Alt+R दबाएँ, फिर NVDA के reading commands से टेक्स्ट पढ़ें। Brief / Standard / Detailed का चयन और moment का शीर्षक भी जाँचें। सिर्फ़ लंबी explanation को पूरा सुनने के लिए assistant की छोटी voice reply पर निर्भर न रहें।

## 8. Timeline और visual जानकारी

Timeline लेक्चर के moments की सूची है। इससे समय, category और शीर्षक देखकर किसी हिस्से पर जा सकते हैं।

### स्क्रीन रीडर से

1. Lecture से Timeline कंट्रोल खोलें।
2. Heading और moment सूची पढ़ें।
3. जरूरत हो तो category filter चुनें।
4. सूची में moment चुनें और उसका preview / विवरण पढ़ें।
5. संबंधित Open कंट्रोल से उस moment को lecture में खोलें। चयन और खोलना अलग चरण हो सकते हैं।

### वॉयस से

| बोलें | काम |
| --- | --- |
| “Show the timeline” | Lecture से Timeline खोलना |
| “Read the list” | Moments के नाम, category और समय सुनना |
| “Show only the graphs” | Graph moments तक filter करना |
| “Show all moments” | सभी categories वापस दिखाना |
| “Go to moment two” | मौजूदा सूची का moment चुनना |
| “What is selected?” | चयनित moment की जानकारी |
| “Open this moment” | चुना moment lecture में खोलना |

Graphs, equations, charts, diagrams, code और tables जैसे filters उपलब्ध सामग्री पर निर्भर हैं। Filter के बाद item number बदल सकता है।

Study mode में Picture और Structured View जैसे tabs सामग्री उपलब्ध होने पर मिलते हैं। Tabs पर फ़ोकस करके Arrow keys, Home और End से चल सकते हैं। “Show me the picture” और “Show the structured view” कमांड भी उपलब्ध संदर्भ में इस्तेमाल करें। “Highlight the important part” के लिए पहचाना हुआ focus region चाहिए।

Accessibility mode में **Accessible visual information** अनुभाग पढ़ें। Structured View की जानकारी न मिलना इस बात का प्रमाण नहीं कि चित्र में कोई विवरण नहीं है। Demo के रूप में चिह्नित उदाहरणों को वास्तविक लेक्चर से निकली जानकारी न समझें।

## 9. अपने वॉयस नोट रिकॉर्ड करना

### 9.1 कीबोर्ड से रिकॉर्ड और सेव

1. Lecture में उस समय पर जाएँ जहाँ नोट चाहिए।
2. Alt+M दबाएँ। लेक्चर और explanation audio pause होंगे।
3. Tone के बाद अपना नोट बोलें।
4. Alt+M दोबारा दबाएँ या Stop recording बटन सक्रिय करें। सामान्य stop पर नोट अपने आप save होता है।
5. Saved status, timestamp और duration सुनें या पढ़ें।
6. शिक्षक का लेक्चर paused रहेगा। आगे सुनने के लिए Alt+P दबाएँ।

### 9.2 वॉयस से शुरू करना

Alt+T से “Record a voice note” बोलें। रिकॉर्डिंग शुरू होने की tone के बाद नोट बोलें। यह **toggle** है: वही action दोबारा होने पर नोट बंद होता है। लेकिन नोट की रिकॉर्डिंग के दौरान command microphone साथ नहीं चल सकता। इसलिए stop के लिए Alt+M या Stop recording बटन सबसे स्पष्ट तरीका है। Talk दबाने पर पहले note recording रुक सकती है; अगला command देने के लिए अलग turn शुरू करें।

आधे सेकंड से छोटी रिकॉर्डिंग reject हो सकती है। अधिकतम अवधि 5 मिनट है; सीमा के पास चेतावनी आती है। Silent recording पर चेतावनी मिल सकती है। रिकॉर्डिंग stop होने के बाद saved संदेश की जाँच करें।

### 9.3 Draft या save error

रिकॉर्डिंग के बीच navigation, microphone disconnect या save failure होने पर Save, Retry, Discard या Stay जैसे विकल्प आ सकते हैं।

- बचाना हो तो Save / Retry चुनें। Draft उपलब्ध हो तो “Save my recording draft” कह सकते हैं।
- हटाना हो तो Discard चुनें या “Discard my voice note draft” कहें।
- सिर्फ़ memory में मौजूद unsaved draft को ऐप बंद करने के बाद सुरक्षित न मानें। पहले save का परिणाम जाँचें।

### 9.4 पुराने नोट खोलना और हटाना

Voice Notes स्क्रीन पर Tab से note controls चलाएँ या ये कमांड दें:

| बोलें | काम |
| --- | --- |
| “Open my voice notes” | नोट सूची खोलना |
| “Play voice note two” | मौजूदा सूची का दूसरा नोट सुनना; दोबारा वही action उसे रोक सकता है |
| “Stop my voice note audio” | चल रहा नोट रोकना |
| “Open voice note two at its time” | नोट के समय पर लेक्चर paused खोलना |
| “Open explanation one attached to note two” | सूची में जुड़ी पहली उपलब्ध explanation खोलना और सुनना |
| “Delete voice note two” | उस नोट को हटाने की confirmation खोलना |
| “Cancel voice note deletion” | नोट रखना |
| “Confirm voice note deletion” | बताए हुए नोट को स्थायी रूप से हटाना |

Confirm करने से पहले note का नाम और समय जाँचें। Recording हटाने को undo उपलब्ध मानकर न चलें।

## 10. Bookmarks और Library

### 10.1 मौजूदा explanation bookmark करना

Lecture में Bookmark कंट्रोल पर Enter दबाएँ या “Bookmark this explanation” बोलें। Explanation अभी save हो रही हो तो पूरा होने दें। यह toggle है: दोबारा वही action bookmark हटा सकता है। अंतिम स्थिति या “Bookmark removed” जैसे संदेश सुनें।

### 10.2 Saved सामग्री इस्तेमाल करना

1. “Open my library” या “Open bookmarks” बोलें; कीबोर्ड से navigation बटन भी इस्तेमाल कर सकते हैं।
2. “Read my saved items” से सूची सुनें। “Read more” आगे के entries पढ़ सकता है; यह अगला page खोलने से अलग है।
3. शीर्षक, moment, समय और depth जाँचें।
4. “Read saved item two” से saved explanation का टेक्स्ट बिना auto-play खोलेँ।
5. “Play the audio of the first item” से saved explanation सुनें।

अतिरिक्त कमांड:

- “Show recent explanations” — Library की recent explanations view।
- “Filter saved items by calculus” — विषय / lecture title के अनुसार filter।
- “Clear the saved items filter” — filter साफ़ करना।
- “Next page of saved items” / “Previous page of saved items” — उपलब्ध अगला / पिछला page।
- “Retry loading saved content” — load failure पर फिर प्रयास।
- “Import my browser saves” — उपलब्ध पुरानी browser saves को local backend storage में import करना।

### 10.3 Saved entry हटाना

“Delete the first bookmark” जैसे कमांड या Remove बटन इस्तेमाल कर सकते हैं। हटाने का लक्ष्य ध्यान से सुनें। वॉयस layer “Yes / No” पूछ सकती है; दिखाई देने वाले confirmation के लिए “Confirm saved item removal” या “Cancel saved item removal” भी उपलब्ध हो सकते हैं। दोनों तरीकों का prompt क्रम बिल्कुल एक जैसा होना जरूरी नहीं है।

Library से lecture हटाने पर bookmarks और recordings रखे जाते हैं। Bookmark हटाना और व्यक्तिगत recording हटाना अलग काम हैं।

Saved version में कोई depth नहीं हो तो Process / Later का निर्णय आएगा। Saved सामग्री होना सभी depths और audio के पहले से उपलब्ध होने की गारंटी नहीं है।

## 11. लेक्चर से सवाल पूछना

### स्क्रीन रीडर से

1. सही lecture खोलें।
2. Ask this lecture अनुभाग पर जाएँ।
3. सवाल लिखें और संबंधित submit बटन सक्रिय करें। सवाल खाली न हो और 500 characters से अधिक न हो।
4. उत्तर और उपलब्ध source / evidence पढ़ें।
5. Timestamp link खोलें तो लेक्चर उसी समय पर paused खुल सकता है; शिक्षक को सुनने के लिए play करें।

### वॉयस से

“What does the axis on this graph represent?” जैसा छोटा सवाल बोलें। ऐप उपलब्ध lecture context से उत्तर लेने की कोशिश करेगा। कोई lecture खुला न हो तो पहले उसे खोलें।

ऐप चलाने का सवाल अलग है: “How do I bookmark this?” से निर्देश मिलते हैं। इसे bookmark करने की सीधी कमांड न समझें। उत्तर में अपर्याप्त संदर्भ बताया जाए तो संबंधित moment खोलकर दोबारा अधिक स्पष्ट सवाल पूछें।

## 12. Practice और Assessment quiz

Quiz तभी इस्तेमाल करें जब उस lecture के लिए सुविधा उपलब्ध हो। इसकी उपलब्धता backend और feature settings पर भी निर्भर है।

### 12.1 Quiz शुरू करना

Lecture में Practice this lecture बटन सक्रिय करें या “Start the practice quiz” बोलें। इससे setup स्क्रीन खुलती है। फिर:

- “Start the practice questions” से सामान्य Practice set माँगें। मौजूदा default **6 questions** है।
- “Start an assessment with eight questions” से mode और count स्पष्ट बोल सकते हैं। अनुमत count 5–10 है।
- कीबोर्ड से mode और count चुनकर generate बटन भी चला सकते हैं। Voice start में mode / count न बताने पर voice defaults लगते हैं; केवल setup form में बदला selection उसी तरह इस्तेमाल होगा, यह न मानें।
- पुराना attempt मिले तो Resume कंट्रोल पढ़कर चुनें। Resume के लिए dedicated voice command पर निर्भर न रहें।

### 12.2 सवालों का उत्तर

स्क्रीन रीडर से question heading पढ़ें। Answer radio group में Arrow keys से विकल्प चुनें। फिर Check answer, Hint, Next, Previous या Review का बटन Enter से चलाएँ।

| बोलें | काम / सीमा |
| --- | --- |
| “Read the question” | मौजूदा सवाल और choices सुनना |
| “Option two” | दूसरा answer चुनना; इससे check या submit नहीं होता |
| “Check my answer” | Practice में चुना answer जाँचना |
| “Give me a hint” | Practice में उपलब्ध hint |
| “Next question” | अगला सवाल, यदि मौजूद हो; वॉयस navigation पर सवाल पढ़ा जाता है |
| “Previous question” | पिछला सवाल, यदि मौजूद हो |
| “Review my answers” | Submission से पहले review खोलना |
| “Submit my answers” | Review स्क्रीन से attempt submit करना |
| “Go back to the lecture” | Setup या results से lecture लौटना |

Practice का score पहले **checked answer** पर आधारित है। केवल विकल्प चुनकर check न करना unanswered गिना जा सकता है। Assessment में प्रति सवाल check / hint नहीं होता; submit पर अंतिम selection गिना जाता है।

Flag, Reveal answer, पुराने attempt का Resume और सीधे किसी question number पर jump के लिए सामान्य UI controls इस्तेमाल करें जहाँ उपलब्ध हों। उत्तर देते समय “Go back to the lecture” की dedicated quiz command उपलब्ध नहीं रहती; सामान्य navigation controls का उपयोग करें।

### 12.3 Submit से पहले

1. Review पर unanswered और flagged items पढ़ें।
2. जवाब सुधारना हो तो संबंधित कंट्रोल से सवाल पर लौटें।
3. Submit सक्रिय करें। Unanswered warning मिले तो उसे पढ़कर निर्णय लें।
4. Submit के बाद answers बदले नहीं जा सकते।
5. Results में score, feedback और source timestamps पढ़ें। Save / conflict error आए तो सफलता मानकर स्क्रीन बंद न करें; उपलब्ध recovery instructions अपनाएँ।

## 13. रूप बदलना, export और Help

### Appearance

- Settings बटन या “Open settings” से theme controls खोलें। Paper, Night या Sepia चुनें। “Switch to night mode” जैसी सीधी कमांड भी दे सकते हैं।
- Escape या “Close settings” से panel बंद करें।
- Lecture में “Switch to accessibility mode” / “Switch to study mode” बोलें या Presentation mode controls चुनें।
- जहाँ display controls उपलब्ध हों, “Make the text bigger” / “Make the text smaller” और “Turn on high contrast” इस्तेमाल कर सकते हैं। Contrast action toggle है; दोहराने से बंद हो सकता है। अंतिम स्थिति सुनें।
- Browser zoom के लिए Ctrl++ / Ctrl+- और reset के लिए Ctrl+0 भी उपयोगी हैं।

### Export

Lecture की चुनी explanation पर “Export this as a PDF” या “Download the audio” बोलें। संबंधित visible download controls भी उपलब्ध होने पर इस्तेमाल करें। सही moment और depth पहले चुनें। Audio तैयार न हो तो उसके download को उपलब्ध न मानें। Browser / Windows का save dialog खुले तो उसे कीबोर्ड और NVDA से अलग पूरा करें।

### Help dialog

“Help” बोलने पर छोटा spoken command guide मिलता है। पूरा Help dialog खोलने के लिए **Help बटन** पर Enter दबाएँ।

Dialog खुलने पर फ़ोकस उसके अंदर जाता है। Tab / Shift+Tab से उसी के controls पर चलें। Escape या Close से बंद करें। पीछे की स्क्रीन का Talk और shortcuts इस दौरान बंद रहते हैं। Help खोलने से paused audio बंद करने के बाद अपने आप शुरू नहीं होता; जरूरत हो तो play करें। Note recording चल रही हो तो पहले Alt+M से रोकें, फिर Help खोलें।

## 14. कौन-सी आवाज़ किसकी है?

स्क्रीन बदलने पर Synapse उसका परिचय बोल सकता है। Status और error updates भी app speech में सुनाए जा सकते हैं, इसलिए ऐप की आवाज़ और NVDA दोनों सुनाई दे सकते हैं।

- **Synapse की speech:** assistant reply, सवाल, screen introduction और कुछ status messages। Backend TTS विफल हो तो उपलब्ध system speech, फिर screen-reader live region का fallback है।
- **NVDA की speech:** फ़ोकस किया heading, button, radio state, dialog, text और live updates।
- **शिक्षक / explanation / note audio:** अपने अलग playback controls से चलता है।

Screen-reader-only feedback एक stored preference के रूप में समर्थित है, लेकिन अभी इसे बदलने का सामान्य Settings या voice selector नहीं है। सिर्फ़ guide पढ़कर ऐसा selector मिलने की उम्मीद न रखें।

NVDA कभी “selected”, “pressed”, “expanded” या “disabled” कहता है। यह कंट्रोल की state है। हर सफल काम पर अलग “Success” sentence आना जरूरी नहीं। लंबा टेक्स्ट पढ़ने के लिए NVDA और पेज का readable content इस्तेमाल करें; छोटी assistant replies पूरी explanation की जगह नहीं लेतीं।

## 15. समस्या आने पर क्या करें

| समस्या / संदेश | अगला कदम |
| --- | --- |
| Alt+T से कुछ नहीं होता | Help बंद करें; Talk की उपलब्धता जाँचें; ऐप पर फ़ोकस लाएँ; Talk बटन Enter से आजमाएँ |
| Space से गलत काम या कोई प्रतिक्रिया नहीं | टेक्स्ट फ़ील्ड / NVDA mode जाँचें; वॉयस के लिए Alt+T और buttons के लिए Enter इस्तेमाल करें |
| Microphone permission blocked | Site permission और Windows Settings → Privacy & security → Microphone में संबंधित app / desktop access जाँचें; फिर कोशिश करें |
| No microphone found | सही device जोड़ें / enable करें और Windows input device जाँचें |
| Microphone busy | दूसरी recording / calling app में microphone छोड़ें; note recording रोकें |
| “I didn't hear anything” | Tone के बाद बोलें; सही microphone और input level जाँचें |
| “I didn't catch that” | थोड़ा पूरा और स्पष्ट वाक्य बोलें |
| “Please answer in English” | मौजूदा विकल्प या कमांड अंग्रेज़ी में बोलें |
| “That isn't available on this screen” | Where am I / Help पूछें; सही स्क्रीन खोलें या visible control इस्तेमाल करें |
| “Which one?” | मौजूदा सूची का नंबर बोलें, जैसे “The second one” |
| “The list has changed” | सूची दोबारा पढ़ें और item फिर चुनें |
| “There is nothing to confirm” | सक्रिय confirmation नहीं है; मूल action फिर स्पष्ट चुनें |
| Server unavailable | Text / keyboard controls से उपलब्ध काम करें; backend connection जाँचें; सेवा लौटने पर voice retry करें |
| Explanation audio नहीं चलती | Alt+R से टेक्स्ट पढ़ें; Listen / Retry उपलब्ध हो तो आजमाएँ या Resume करें |
| गलत moment का संदर्भ लग रहा है | Title, timestamp और depth जाँचें; सही moment खोलकर फिर request करें |
| Recording save नहीं हुई | Draft / Retry संदेश पढ़ें; save सफल होने तक ऐप बंद न करें |
| बोलते समय teacher / NVDA भी सुनाई दे | Headphones, NVDA के लिए Ctrl और teacher के लिए Pause इस्तेमाल करें |
| “Sorry, that didn't work” | लक्ष्य और prerequisites जाँचें; उसी काम का visible control आजमाएँ; error text पढ़ें |

वॉयस पहचानने और कार्रवाई पूरी होने में अंतर है। “Okay” सुनने के बाद भी सही चयन, स्क्रीन या playback जाँचना उपयोगी है।

## 16. पहली पढ़ाई का पूरा उदाहरण

### स्क्रीन रीडर और कीबोर्ड से

1. NVDA शुरू करें और Synapse खोलें।
2. अकाउंट में sign in करें; Welcome पर Accessibility mode चुनें।
3. Home पर Alt+S दबाएँ, `electric fields` लिखें और search करें।
4. सही result का title पढ़ें और उसका open control सक्रिय करें।
5. Lecture खुलने पर Alt+P से शिक्षक को सुनें।
6. Explanation का सवाल आए तो Tab से Listen बटन चुनें और Enter दबाएँ।
7. टेक्स्ट पढ़ना हो तो audio रोककर Alt+R दबाएँ और NVDA से पढ़ें।
8. Explanation स्पष्ट थी तो सवाल के Yes बटन पर Enter दबाएँ; दूसरी depth चाहिए तो No चुनें।
9. अपना नोट जोड़ने के लिए Alt+M, tone के बाद बोलें, फिर Alt+M दबाएँ। Saved संदेश जाँचें।
10. Alt+P से आगे पढ़ें। जरूरत पर Bookmark बटन इस्तेमाल करें।
11. बाद में Library, Bookmarks या Voice Notes खोलकर अपना काम दोबारा पढ़ें / सुनें।

### वॉयस और स्क्रीन रीडर को साथ इस्तेमाल करके

नीचे हर नई कमांड से पहले Alt+T दबाएँ। ऐप ने अपने आप Listening शुरू किया हो तो tone के बाद सीधे जवाब दें। Password और unsupported controls के लिए कीबोर्ड इस्तेमाल करें।

1. “Go to the home screen.”
2. “Search for lectures about electric fields.”
3. “Read the results.”
4. “Choose the first result.”
5. “Start learning.”
6. Lecture खुलने के बाद “Play the lecture.”
7. Explanation का सवाल और tone आने पर “Listen.”
8. Explanation के बाद “Yes”, या “No” और फिर पेश किए विकल्प में से “Detailed.”
9. “Bookmark this explanation.” अंतिम bookmark स्थिति सुनें।
10. “Record a voice note.” रिकॉर्डिंग की tone के बाद अपना नोट बोलें और Alt+M से stop करें।
11. Saved संदेश के बाद नए voice turn में “Play the lecture.”
12. बाद में “Open bookmarks”, फिर “Read saved item one”; NVDA से खुले टेक्स्ट को पढ़ें।

## 17. मौजूदा सीमाएँ और संदर्भ

- वॉयस इनपुट अंग्रेज़ी में है। हिंदी guide होने का अर्थ हिंदी command support नहीं है।
- Password टाइप करना पड़ता है। Sign out, processing cancellation और पूरा Help dialog जैसे कामों के लिए संबंधित UI controls इस्तेमाल करें।
- हर command हर screen या हर state में उपलब्ध नहीं है। कोई सूची खाली हो, explanation तैयार न हो या confirmation बदल गया हो तो पहले वही स्थिति ठीक करें।
- Library, bookmarks और recordings इस local installation / account के storage पर निर्भर हैं। दूसरे कंप्यूटर पर अपने आप sync होने की उम्मीद न करें। Browser की बची पुरानी सामग्री के लिए import / recovery विकल्प अलग हो सकते हैं।
- AI explanation और quiz उत्तर को जरूरत पड़ने पर शिक्षक और source timestamp से जाँचें, खासकर संख्या, समीकरण, labels और दिशा जैसी बातों को।

यह guide 16 नवंबर के मौजूदा source व्यवहार को प्राथमिकता देती है। पुराने reference में manual follow-up, सीमित account voice flow या fixed 5-question quiz का वर्णन मिल सकता है; वर्तमान code में auto-listen, username / Welcome actions और quiz mode / count support मौजूद हैं।

संबंधित सामग्री:

- [अंग्रेज़ी voice command और announcement reference](synapse-web-voice-reference.md)
- [पूरी manual verification guide](synapse-web-manual-test-guide.md)
- [असली keyboard shortcuts का स्रोत](../synapse-web/src/lib/shortcuts.js)
- [Talk, Space और auto-listen व्यवहार](../synapse-web/src/voice/VoiceControl.jsx)
- [Account form](../synapse-web/src/components/AccountForm.jsx)
- [Quiz voice handlers](../synapse-web/src/components/QuizScreen.jsx)
