# GPT4 PROMPT
'''
I want you to act as  a action analysis expert. I will give you a action list and you task is to give me 3 sentences to describe the each action. 
Your description must accurately revolve around this action and be as objective, detailed and diverse as possbile. 
In addition, the subject of your description is a human action executed in a earthquake video clip. 
The word list is ["aiming weapon, arguing, arresting, burning/setting fire, carrying"],Please follow the following dictionary format for your output:
{word1: [description1, description2, description3], word2: [description1, description2, description3],...}
'''
_LABEL_EXTEND = {
  "aiming weapon": [
    "A person is seen steadying their posture amidst trembling grounds, focusing intently on a target, weapon in hand, ready to defend or attack based on the perceived threat.",
    "The individual's grip tightens on the weapon, eyes narrowing as they calculate the distance and movement, compensating for the unstable environment caused by the earthquake.",
    "Amidst the chaos of the shaking surroundings, the figure maintains a controlled breath, aiming the weapon with precision, demonstrating a high level of training and composure under pressure."
  ],
  "arguing": [
    "Two individuals are captured in a heated exchange, their voices raised over the rumbling earth, each trying to convey their point with urgent gesticulations, reflecting the stress of the crisis situation.",
    "Amidst the debris and panic, a person is seen vehemently arguing with a rescuer, possibly over safety procedures or directions, their face marked by frustration and fear, highlighting the communication breakdown in emergencies.",
    "The video captures a moment of intense argument between group members, as they disagree on the safest course of action, their animated expressions and body language indicating the high stakes of their decision-making process."
  ],
  "arresting": [
    "In the midst of chaos, law enforcement personnel are shown executing an arrest, carefully maneuvering the suspect to the ground with practiced efficiency, even as aftershocks threaten their balance.",
    "The video shows an officer handcuffing a person against a backdrop of crumbling structures, the act of arresting juxtaposed with the imperative of immediate disaster response, highlighting the complexity of maintaining order during natural catastrophes.",
    "A sequence depicts the swift and methodical arrest of an individual by authorities, who despite the ongoing tremors, ensure the safety of all involved, underscoring the challenge of upholding law amidst disaster."
  ],
  "burning/setting fire": [
    "A scene unfolds where a person is desperately trying to set a signal fire, striking matches against the wind, the small flames flickering against the backdrop of destroyed buildings, signifying a call for help.",
    "The aftermath of the earthquake reveals a fire deliberately set to clear debris, with individuals working around the blaze, carefully managing its spread to prevent further damage, showcasing human resilience and adaptability.",
    "In an act of desperation or destruction, a fire rages through a compromised structure, possibly set by survivors to attract rescue or by nefarious actors exploiting the chaos, illustrating the dual nature of human responses to disaster."
  ],
  "carrying": [
    "A figure is seen laboriously carrying an injured person over their shoulders, navigating through rubble-strewn streets, a testament to human strength and compassion in the face of disaster.",
    "The video captures individuals forming a human chain to carry supplies across a collapsed bridge, their coordinated effort symbolizing community solidarity and the urgent need for survival resources.",
    "Amidst the devastation, a person is shown carrying personal belongings in a makeshift backpack, moving away from the epicenter, their slow, determined pace reflecting the gravity of the situation and the instinct to preserve life and memories."
  ],
  "carrying casualty": [
    "A rescuer is seen carefully lifting a casualty with evident injuries, maneuvering through the debris with focused determination, highlighting the urgency and compassion involved in search and rescue operations.",
    "The video captures a tense moment where individuals work together to carry a stretcher, navigating through narrow, rubble-filled paths, their coordinated effort a poignant reminder of the immediate aftermath of disaster.",
    "Amid the dust and aftershocks, a person is depicted using an improvised sling to support an injured companion, their slow but steady progress towards safety illustrating the resilience and adaptability required in crisis situations."
  ],
  "chanting/cheering": [
    "A group of survivors, gathered amidst the ruins, raise their voices in a unified chant, their collective spirit serving as a beacon of hope and solidarity in the face of overwhelming destruction.",
    "The clip shows an emotional scene where people, surrounded by devastation, spontaneously burst into cheers, celebrating the successful rescue of victims trapped under the rubble, symbolizing the triumph of life over disaster.",
    "In a moment of communal defiance against despair, individuals are captured chanting motivational slogans, their resonant voices echoing through the desolate landscape, embodying the human capacity to find strength in unity."
  ],
  "chasing": [
    "The footage reveals a frantic chase as emergency responders rush towards the epicenter of the quake, dodging falling debris and navigating chaotic streets, embodying the urgent race against time to save lives.",
    "A person is seen running after a stray animal amidst the tremors, their determined pursuit through hazardous terrain highlighting the instinctive human response to protect all forms of life during crises.",
    "In an unexpected turn, the video captures a scene of citizens chasing down a looter, their swift coordination and resolve reflecting the community's stand against exploitation of the disaster's chaos."
  ],
  "clapping": [
    "As a survivor is pulled from the rubble, the crowd around erupts into spontaneous clapping, a universal gesture of relief and gratitude, echoing amidst the sounds of ongoing rescue operations.",
    "The video features a poignant moment where rescuers and survivors alike pause to clap for a moment of silence in honor of the lives lost, a shared expression of mourning and respect amid the tragedy.",
    "A group of aid workers receives a warm welcome as they arrive at the disaster site, with locals clapping in appreciation of their support, showcasing the communal acknowledgment of assistance and solidarity."
  ],
  "creating barricade": [
    "Individuals are shown hastily assembling a barricade using available materials like rubble and overturned vehicles, aiming to control movement or protect a vulnerable area from further damage or looting.",
    "The clip details the strategic placement of barriers to demarcate safe zones around unstable buildings, with people working together to reinforce these makeshift barricades against potential aftershocks.",
    "In an effort to manage the chaos, survivors and emergency teams collaborate to create barricades, directing the flow of evacuees towards aid stations, an organized response highlighting the importance of immediate urban planning in disaster scenarios."
  ],
  "dancing": [
    "In a moment of surreal defiance against the backdrop of devastation, a person is seen dancing amidst the ruins, their movements expressive of hope and resilience, a stark contrast to the surrounding despair.",
    "The video captures an individual engaging in a traditional dance, perhaps in an attempt to preserve a sense of normalcy and cultural identity amidst the chaos, their rhythmic movements serving as a reminder of the enduring human spirit.",
    "Surrounded by emergency crews and amidst ongoing aftershocks, a group of people dance in a circle, their linked hands and synchronized steps symbolizing unity and the collective will to overcome the tragedy."
  ],
  "destroying": [
    "Amidst the rubble, a figure is seen purposefully destroying what remains of an unstable structure, their actions driven by the necessity to prevent further harm and to clear the way for rescue and rebuilding efforts.",
    "The clip shows a person in a fit of anger or desperation, taking out their frustration on inanimate objects, their actions reflecting the psychological toll and sense of loss induced by the earthquake.",
    "A machine operated by a rescue worker moves through the debris, methodically destroying parts of buildings that pose a risk of collapse, illustrating the delicate balance between destruction and safety in disaster response."
  ],
  "extinguishing fire": [
    "In the immediate aftermath of the earthquake, individuals are shown forming a human chain to pass buckets of water, working tirelessly to extinguish a fire that threatens to consume what is left of a community building.",
    "A lone figure battles a blaze with a fire extinguisher, the small but determined effort highlighting the personal bravery and quick action that can save lives and property in the wake of natural disasters.",
    "Emergency responders move with precision and urgency, deploying hoses and water jets to combat a fire sparked by the earthquake, their coordinated efforts a testament to the critical role of firefighting in disaster scenarios."
  ],
  "fighting": [
    "The video reveals a tense confrontation between two individuals, their physical altercation possibly ignited by stress, fear, or disputes over resources in the chaotic environment following the earthquake.",
    "Amidst the confusion and panic, a group is shown clashing over access to emergency supplies, their physical engagement a stark reminder of the desperation and social strain that can arise in the wake of disaster.",
    "Security forces intervene in a skirmish among survivors, their efforts to restore order and prevent violence underscoring the heightened tensions and challenges of maintaining peace in a disaster-stricken area."
  ],
  "guarding": [
    "A person stands vigilant at the entrance of a makeshift shelter, their posture firm as they guard the vulnerable against potential threats, embodying the human instinct to protect community and kin in times of crisis.",
    "The video shows members of the local community taking turns to guard salvaged possessions and supplies, their watchful eyes and coordinated efforts ensuring the safety of their collective resources.",
    "Armed forces are depicted establishing a perimeter around critical infrastructure, their presence a deterrent against looting and a measure of security to facilitate the orderly distribution of aid and the continuation of rescue operations."
  ],
  "hitting an object/smashing": [
    "A person is captured in the act of hitting a partially collapsed wall with a sledgehammer, each strike aimed at creating a passage for rescue or salvage efforts, highlighting the physical exertion involved in disaster recovery.",
    "The video shows an individual frantically smashing a window to access trapped survivors inside a vehicle, their actions driven by urgency and desperation to save lives amidst the chaos of the earthquake.",
    "Amid the rubble, a figure is seen pounding on a sturdy object, possibly to signal their location to rescuers or to vent frustration, embodying the intense emotional and physical responses elicited by such catastrophic events."
  ],
  "holding a burning stick": [
    "In the dim aftermath of the quake, a person is shown holding a burning stick aloft, using it as a makeshift torch to illuminate a path through the darkness, their face set in determination to find a way to safety.",
    "The clip captures a solemn scene of an individual holding a burning stick, perhaps as part of a vigil or commemoration for those lost, symbolizing both mourning and the resilience of the human spirit.",
    "A survivor navigates the debris-strewn streets, holding a burning stick, using the feeble light to search for missing loved ones or to signal for help, a poignant image of hope amidst despair."
  ],
  "holding flag": [
    "Amidst the devastation, a figure stands with a flag held high, the emblem fluttering as a sign of national solidarity or a call to rally support, conveying a powerful message of unity and defiance in the face of adversity.",
    "The video shows an individual moving through the affected area, holding a flag that represents a cause or community, their action turning the flag into a beacon of hope and a symbol of collective identity during the crisis.",
    "In a quiet moment of reflection, a person is seen holding a flag, perhaps marking the site of a significant loss or claiming a space for communal gathering, imbuing the scene with a sense of purpose and remembrance."
  ],
  "holding hands": [
    "The footage reveals a chain of people holding hands, forming a human link through the debris as they evacuate the danger zone, a testament to the strength found in solidarity and mutual support during emergencies.",
    "A family is depicted holding hands tightly as they navigate the uncertain terrain, their physical connection a source of comfort and a symbol of their unbreakable bond in the face of the earthquake's chaos.",
    "Survivors gather in a circle, holding hands in a moment of silence for the lives lost, their united front serving as a powerful expression of community resilience and collective mourning."
  ],
  "holding signage": [
    "A person is seen holding a sign with messages of gratitude towards the rescuers, standing amidst the ruins, their gesture a poignant reminder of the human capacity for gratitude and recognition even in the darkest times.",
    "The video captures an individual holding up a sign seeking information about missing relatives, a silent plea made visible against the backdrop of ongoing search and recovery operations, illustrating the desperate search for connection amid catastrophe.",
    "Amid the recovery efforts, a group holds signage offering free services to affected individuals, from medical aid to shelter, their proactive stance highlighting the community's collective effort to rebuild and support each other."
  ],
  "holding weapon": [
    "Amid the chaos of the aftermath, a security officer is seen holding a weapon, vigilant and ready to maintain order, their stance reflecting the tension and the need for protection in the wake of disaster.",
    "The video captures a civilian holding a weapon, perhaps found among the debris, their expression one of fear and determination to defend themselves and their loved ones from potential looters or other threats.",
    "A rescue worker holds a weapon, not as a threat, but as a precautionary measure to ensure the safety of the rescue team and survivors, symbolizing the complex challenges faced during disaster response efforts."
  ],
  "hugging": [
    "In a powerful display of human connection, two survivors are shown embracing tightly, their hug a silent expression of relief, support, and shared grief amid the surrounding devastation.",
    "The clip captures a moment of reunion, where individuals who were separated during the earthquake come together in a prolonged hug, a poignant reminder of the importance of human relationships in times of crisis.",
    "Amidst the rubble, a rescue worker and a survivor share a brief hug, a gesture of comfort and gratitude that transcends words, highlighting the emotional intensity of search and rescue operations."
  ],
  "injured": [
    "An injured individual is depicted receiving first aid from emergency responders, the seriousness of their wounds underscored by the urgency and care with which they are treated, reflecting the immediate medical challenges following an earthquake.",
    "The video shows a person limping through the debris, their movements hindered by visible injuries, a testament to the physical toll extracted by the quake and the resilience of those affected.",
    "Lying amidst the rubble, an injured survivor is shown signaling for help, their condition a stark representation of the human cost of natural disasters and the critical need for timely assistance."
  ],
  "kneeling": [
    "A figure is seen kneeling in prayer or reflection amid the destruction, their solitary form a poignant symbol of hope, faith, and the search for solace in the aftermath of the earthquake.",
    "The video captures a rescue worker kneeling beside a collapsed structure, carefully listening for signs of life, their posture indicative of the focused determination inherent in the search for survivors.",
    "In a moment of exhaustion and despair, a survivor kneels on the ground, their body language conveying the overwhelming impact of the disaster and the human struggle to comprehend and confront such loss."
  ],
  "painting": [
    "Amid the disarray, an individual is shown painting a mural on a standing wall, their artwork transforming a symbol of destruction into a message of hope and resilience for the community.",
    "The clip features a person painting signs onto makeshift boards, directing survivors to aid stations or safe zones, illustrating the use of art as a practical tool for communication in crisis situations.",
    "A survivor uses painting as a form of therapy, capturing scenes of the earthquake's aftermath on canvas, their brushstrokes a testament to the cathartic power of creativity in processing trauma and loss."
  ],
  "pinning": [
    "A security officer is seen pinning down a person amidst the chaos, possibly to prevent harm or disorder, their actions reflective of the need to maintain control in the unstable environment following the earthquake.",
    "The video captures a moment of rescue, where a volunteer pins down a piece of unstable debris, securing a safe path for survivors to evacuate, illustrating the physical challenges and quick thinking required in disaster response.",
    "In the midst of aftershocks, an individual is depicted pinning a notice to a remaining wall, their message offering directions or information, highlighting the use of makeshift methods to communicate in times of crisis."
  ],
  "playing instrument": [
    "Against a backdrop of destruction, a person is shown playing a musical instrument, their melody a defiant beacon of hope and a means of solace for both the player and the onlookers, symbolizing the enduring human spirit.",
    "The clip features a group of survivors gathered around, listening to an individual playing an instrument, their music creating a temporary escape from the surrounding devastation, fostering a sense of community and resilience.",
    "A musician is captured playing a somber tune amidst the rubble, their performance an act of remembrance for those lost, and a poignant reminder of the role of art in healing and uniting people during times of sorrow."
  ],
  "praying": [
    "In a quiet corner of the disaster zone, an individual is seen kneeling and praying, their silent invocation a personal refuge and a source of strength in the face of overwhelming loss and uncertainty.",
    "The video shows a group of people, from different faiths, coming together to pray, their unified act of worship demonstrating the unifying power of faith in bringing comfort and hope to distressed communities.",
    "Amidst the rubble, a survivor stands with eyes closed and hands clasped in prayer, their moment of spiritual solace amidst chaos serving as a testament to the search for meaning and guidance in the wake of tragedy."
  ],
  "pulling barricade": [
    "A team of emergency responders is seen pulling a barricade to block off a dangerous area, their coordinated effort crucial in preventing further injuries and ensuring the safety of the ongoing rescue operations.",
    "The clip captures residents working together to pull a barricade across a street, their action a community-led initiative to protect their neighborhood from potential looting or unauthorized entry in the aftermath of the earthquake.",
    "In an effort to facilitate access for rescue vehicles, volunteers are shown pulling away barricades that had been hastily erected, their determination to clear paths underscoring the collective human effort to respond to and recover from disaster."
  ],
  "punching": [
    "The footage reveals a person punching through a thin wall or barrier in an attempt to reach trapped survivors, their physical exertion emblematic of the desperate and immediate actions taken to save lives.",
    "In a display of frustration and anger, an individual is captured punching a damaged object, their emotional response to the situation reflecting the intense stress and helplessness felt by many in the aftermath of the earthquake.",
    "A rescue worker is shown punching a code into a secure lock, their rapid movements indicative of the urgency to access emergency supplies or equipment needed for the disaster response efforts."
  ],
  "pushing": [
    "A person is shown pushing against a heavy obstacle, their efforts aimed at clearing a path for emergency responders or fellow survivors, highlighting the physical challenges and determination involved in disaster response.",
    "The video captures a group of people pushing a stalled vehicle to the side of the road, making room for rescue and medical teams to pass, showcasing the collective effort to manage the aftermath efficiently.",
    "Amid the tremors, an individual pushes through a crowd, desperately trying to reach a loved one or to deliver aid, illustrating the urgency and panic that can drive human actions in the face of crisis."
  ],
  "raising fist": [
    "In a moment of solidarity or protest, a survivor is seen raising their fist into the air, a powerful gesture symbolizing defiance, resilience, or the demand for change in the wake of the disaster.",
    "The footage includes a poignant scene where, amidst the devastation, a person raises their fist as a sign of victory after being rescued, embodying the triumph of survival against the odds.",
    "A group of volunteers, standing in the ruins, raise their fists together in a moment of commitment to rebuilding efforts, their unified stance serving as a testament to community strength and determination."
  ],
  "raising hands": [
    "Survivors are depicted raising their hands in a gesture of surrender or to signal for help, their open palms visible against the backdrop of destruction, highlighting their vulnerability and the immediate need for assistance.",
    "The video shows a crowd raising their hands in applause as rescuers successfully extract a survivor from the rubble, a spontaneous expression of gratitude and relief in the midst of sorrow.",
    "In a makeshift shelter, individuals raise their hands to volunteer for community tasks, demonstrating the willingness to contribute and support each other in the collective recovery process."
  ],
  "recording": [
    "An individual is captured on video recording the aftermath of the earthquake with their phone, documenting the extent of the damage and the ongoing rescue efforts, a testament to the role of citizen journalism in crisis situations.",
    "The clip shows a professional journalist recording a live segment amidst the ruins, their camera panning over the scene to convey the immediate impact of the disaster to viewers around the world, highlighting the importance of media in disaster response.",
    "A person records a personal message in the disaster-stricken area, their video diary serving as a poignant record of their emotions and experiences, preserving a firsthand account of the earthquake's human toll."
  ],
  "reporting live": [
    "A reporter stands in the midst of the devastation, delivering a live update to the audience, their voice steady as they describe the scene, the scale of the destruction, and the ongoing efforts to find survivors.",
    "The footage captures a journalist reporting live, wearing a hard hat and safety gear, their report punctuated by aftershocks or the sound of emergency sirens, providing real-time insights into the chaotic environment.",
    "In an effort to raise awareness and aid, a broadcaster reports live from a temporary relief camp, interviewing survivors and volunteers, their coverage helping to mobilize global support and resources for the affected community."
  ],
  "retreating": [
    "A group of people is seen quickly retreating from a building showing signs of imminent collapse, their rapid movements and alert expressions indicative of the urgent need to find safer ground amidst the aftershocks.",
    "The video captures an individual retreating from the edge of a newly formed fissure in the ground, their cautious backward steps reflecting the unpredictable nature of the terrain after the earthquake.",
    "Emergency responders are shown retreating from a hazardous area marked by gas leaks, their organized withdrawal demonstrating the importance of safety protocols in preventing further casualties during disaster response operations."
  ],
  "running/escaping": [
    "Survivors are depicted running through debris-littered streets, their swift movements driven by the instinctual urge to escape immediate danger, highlighting the chaos and panic that can ensue in the aftermath of an earthquake.",
    "The footage shows children and adults alike escaping from a crumbling school building, their rapid, coordinated evacuation a testament to the effectiveness of disaster preparedness drills in saving lives.",
    "An individual is captured running from an approaching tsunami triggered by the earthquake, their desperate escape emphasizing the compound disasters that can follow seismic events and the critical window for survival."
  ],
  "shooting/firing": [
    "In the context of maintaining order, security forces are shown firing warning shots into the air to deter looters in the disaster-struck area, their actions reflecting the thin line between chaos and control in the aftermath of an earthquake.",
    "The video includes a distressing scene where an individual resorts to shooting at a locked door to gain access to trapped family members, their extreme measure underlining the dire situations faced by people in their fight for survival.",
    "A flare is fired into the sky by a survivor, the bright trail against the dark backdrop serving as a signal for help, illustrating the use of firearms as a means of communication in situations where conventional methods are insufficient."
  ],
  "shouting": [
    "Rescuers are seen shouting instructions to each other over the roar of collapsing buildings, their loud, clear commands crucial for coordinating their efforts to save trapped survivors.",
    "An individual is captured shouting for help from beneath rubble, their voice a beacon for responders in the silent aftermath, highlighting the role of human sounds in the search and rescue process.",
    "The video shows a community leader shouting words of encouragement to the disaster-affected people, their powerful voice cutting through the despair, mobilizing the community towards collective action and hope."
  ],
  "singing": [
    "Amid the devastation, a group of survivors is shown singing a hymn together, their collective melody a form of emotional support and a way to maintain morale among the group, showcasing the healing power of music.",
    "The clip captures a poignant scene of a lone individual singing softly while searching through the debris for personal belongings, their song a personal anthem of resilience and a means of coping with loss.",
    "In a temporary shelter, children are seen singing songs taught in school, their innocent voices bringing a sense of normalcy and comfort to the otherwise tense atmosphere, highlighting the importance of psychological relief activities in disaster recovery."
  ],
  "speaking on stage": [
    "A community leader is seen speaking on a makeshift stage erected amidst the ruins, addressing a crowd of survivors and volunteers, their voice firm and encouraging, aiming to instill hope and provide updates on relief efforts.",
    "In the wake of the earthquake, an aid organization representative speaks on stage during a relief concert, their speech focusing on the importance of solidarity, support, and the ongoing needs of the affected communities.",
    "A survivor, standing on a platform made of debris, shares their harrowing experience with an assembled group, speaking passionately about the moments of terror and the acts of heroism witnessed, serving as a cathartic outlet for collective grief and resilience."
  ],
  "speaking/talking": [
    "Two individuals are captured in a quiet conversation amidst the backdrop of emergency operations, discussing the next steps for their family, their exchange a blend of practicality and mutual support in uncertain times.",
    "Emergency responders communicate efficiently, speaking into their radios to coordinate search and rescue efforts, their clear and concise communication vital for navigating the chaotic aftermath of the quake.",
    "A volunteer speaks to a distressed survivor, offering words of comfort and explaining the process for receiving medical aid and temporary shelter, highlighting the role of clear information and empathy in disaster response."
  ],
  "spraying": [
    "Firefighters are shown spraying water on the smoldering remains of a collapsed building, their efforts aimed at extinguishing fires triggered by the earthquake, a critical action to prevent further damage and loss.",
    "An individual uses a spray can to mark a demolished structure with symbols indicating it has been searched for survivors, a visual language essential for efficient rescue operations amidst widespread destruction.",
    "A volunteer sprays disinfectant in a temporary shelter, part of the health measures to prevent disease outbreaks among the crowded conditions of survivors, illustrating the ongoing public health efforts following the disaster."
  ],
  "stealing/looting": [
    "The video captures a scene of chaos as individuals take advantage of the breakdown in order, seen stealing goods from a damaged store, their actions reflecting the darker side of human behavior in times of crisis.",
    "Amidst the aftermath, a group is observed looting a supply truck, hurriedly grabbing food and water, a desperate act driven by survival instincts but complicating the distribution of aid to those most in need.",
    "A surveillance camera footage shows a person stealthily looting electronics from an abandoned shop, an act of opportunism that detracts from the community's focus on recovery and mutual aid."
  ],
  "throwing object": [
    "A frustrated and angry survivor is seen throwing rubble at a barricade blocking access to a devastated area, their act a manifestation of the immense emotional stress and impotence felt in the face of disaster.",
    "Children, in a moment of innocent play amidst the devastation, throw small stones into a makeshift game, a poignant reminder of the resilience and adaptability of youth even in the most dire circumstances.",
    "A rescue worker throws a rope to colleagues across a gap, a precise and necessary action to secure a path or deliver supplies across obstructed areas, demonstrating the practical challenges and solutions in disaster environments."
  ],
    "walking": [
    "An individual carefully navigates through the debris-strewn streets, their cautious steps and vigilant eyes reflecting the dangers posed by unstable structures and aftershocks, illustrating the precariousness of movement post-disaster.",
    "A group of survivors walks together towards a designated safe zone, their unified pace and shared direction symbolizing hope and the collective will to rebuild in the aftermath of the earthquake.",
    "Rescue workers traverse the affected area, walking from one site to another, their determined strides and focused expressions highlighting the urgency of their mission to find and aid survivors amidst the ruins."
  ],
  "watching": [
    "Onlookers gather at a safe distance, watching anxiously as emergency teams work to extract survivors from a collapsed building, their faces a mix of hope and despair, embodying the shared human experience of tragedy.",
    "A child, perched on the remains of their home, watches the chaotic scene unfold, their quiet observation contrasting with the loud efforts of rescue and recovery, capturing the innocence affected by disaster.",
    "Survivors watch their phones and portable radios for news updates, their attention glued to the screens and speakers, seeking information about relief efforts, missing persons, and the broader impact of the earthquake."
  ],
  "waving flag": [
    "Amidst the rubble, an individual is seen waving a national flag, a symbol of resilience and unity, their action inspiring a sense of patriotism and collective strength among the watchers and fellow survivors.",
    "A volunteer waves a flag to signal a medical team's location, their bright emblem standing out against the grey backdrop of destruction, indicating where aid and assistance can be found.",
    "In the aftermath of the quake, a person on a rooftop waves a makeshift flag, fashioned from clothing, to attract the attention of rescue helicopters, their resourcefulness a beacon of hope in desperate times."
  ],
  "waving flarestick": [
    "As dusk falls over the disaster zone, a figure is visible waving a flare stick, the bright light piercing through the darkness, signaling their location to rescuers in a powerful display of survival instinct.",
    "Emergency personnel use flare sticks to guide evacuees through smoke and debris, their waving motions a critical navigation tool in environments where traditional landmarks are obscured or destroyed.",
    "A stranded survivor waves a flare stick from within the confines of a collapsed structure, their action not only a call for help but a symbol of human resilience in the face of overwhelming odds."
  ],
  "waving hands": [
    "A group of survivors, trapped atop a partially collapsed building, wave their hands frantically to catch the attention of passing drones and helicopters, their gestures a universal sign of distress and urgent need for rescue.",
    "On the side of a damaged road, an individual waves their hands at passing vehicles, seeking to stop someone who can offer transport to safer areas or hospitals, highlighting the immediate needs for mobility and medical care.",
    "Children wave their hands excitedly at a camera crew, their innocent engagement a stark contrast to the tragedy that surrounds them, reminding viewers of the ongoing human stories within disaster zones."
  ]
}

import json
json_str = json.dumps(_LABEL_EXTEND)
with open('label_extend.json', 'w') as f:
    f.write(json_str)
