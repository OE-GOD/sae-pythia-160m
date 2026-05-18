# SAE Feature Inspection — Pythia-160M, layer 6, TopK k=64

- Total alive features: **16,319 / 16,384**
- Each row below shows a feature's top-K max-activating snippets.
- `act` is the SAE feature activation value at that token.
- `target` is the token being scored (decoded).

Use this report to manually identify which features look monosemantic
(snippets share a clear theme) vs polysemantic (mixed concepts).

## Section A — Top 30 features by single-snippet peak activation

These features have the strongest single activations. Often these are 'sharp' features
triggered by specific tokens or patterns.

### Feature #121

Top 10 max-activating snippets:

1. `act=107.64` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
2. `act=102.72` target=`' Astr'` snippet: ` says research published in the journal Monthly Notices of the Royal Astronomical Society.`
3. `act=102.72` target=`' Astr'` snippet: `Category:2011 establishments in ScotlandGuest Blogger: A Christian Astronomer Reflect`
4. `act=102.15` target=`' cr'` snippet: `. So let’s hope the Japanese brand has managed to cram in plenty of`
5. `act=102.07` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
6. `act=101.94` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
7. `act=101.88` target=`'agles'` snippet: ` Bruce Arians was a candidate for the vacant Philadelphia Eagles job, but the`
8. `act=101.78` target=`' cave'` snippet: `feedback, Dickerson said.  Adcock said one caveat of the study`
9. `act=100.88` target=`' Borough'` snippet: ` to you by the Colchester Travel Plan Club, Colchester Borough Council Air Quality Team`
10. `act=100.80` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`

### Feature #6630

Top 10 max-activating snippets:

1. `act=52.59` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`
2. `act=52.40` target=`' initiated'` snippet: `pson, Swope said at the time the study was initiated.  The`
3. `act=51.91` target=`' mobil'` snippet: ` will have been made available by 2010.  Popular mobilisation needed  `
4. `act=51.68` target=`' hydrogen'` snippet: ` closed conformation of AdoHcyase are identified as the hydrogen bonds between the backbone`
5. `act=51.58` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
6. `act=51.57` target=`' caspase'` snippet: ` programmed cell death pathway of the cells from patient group A was caspase-3 and poly`
7. `act=51.41` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
8. `act=51.37` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
9. `act=51.25` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
10. `act=51.10` target=`' lign'` snippet: `intéressé n'a pu être joint. La ligne télé`

### Feature #2255

Top 10 max-activating snippets:

1. `act=44.43` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=44.22` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=44.15` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
4. `act=44.15` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
5. `act=43.98` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
6. `act=43.86` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
7. `act=43.85` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
8. `act=43.79` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
9. `act=43.75` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
10. `act=43.71` target=`'\\n'` snippet: `:1948 births Category:Alumni of Imperial College London Category:Headmas`

### Feature #12520

Top 10 max-activating snippets:

1. `act=36.48` target=`' initiated'` snippet: `pson, Swope said at the time the study was initiated.  The`
2. `act=36.10` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
3. `act=36.02` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
4. `act=35.99` target=`' dismissed'` snippet: ` record on appeal. The State argues defendants' appeal must be dismissed because the trial court`
5. `act=35.96` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
6. `act=35.96` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
7. `act=35.91` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
8. `act=35.87` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
9. `act=35.86` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
10. `act=35.57` target=`'ichi'` snippet: ` Ishizuka joined team ALIVE which is based in Aichi prefecture. `

### Feature #6767

Top 10 max-activating snippets:

1. `act=33.51` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=33.24` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=33.22` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
4. `act=33.22` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
5. `act=33.08` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=33.03` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
7. `act=32.98` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
8. `act=32.97` target=`'\\n\t'` snippet: ` getPadding -  Returns an object with padding on the edges 	 * @prop {`
9. `act=32.97` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
10. `act=32.96` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`

### Feature #6484

Top 10 max-activating snippets:

1. `act=32.50` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=32.48` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=32.48` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=32.21` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=32.14` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
6. `act=32.00` target=`'\r\\n\t\t\t'` snippet: `>  			<xsl:element name="IntValue">  				<xsl:`
7. `act=31.96` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
8. `act=31.94` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
9. `act=31.81` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
10. `act=31.80` target=`'\\n\t\t\t'` snippet: ` -1 			if result.Response() != nil { 				sc = result`

### Feature #15230

Top 10 max-activating snippets:

1. `act=32.11` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=32.08` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=31.92` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
4. `act=31.88` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
5. `act=31.75` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
6. `act=31.74` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
7. `act=31.74` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
8. `act=31.73` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
9. `act=31.71` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
10. `act=31.68` target=`'\\n'` snippet: `regularly-how-hard-can-it-be ====== soph`

### Feature #5747

Top 10 max-activating snippets:

1. `act=30.75` target=`' faced'` snippet: ` a combination of both?Those are just some of the questions faced by today’s`
2. `act=30.44` target=`' on'` snippet: ` we were a smaller organization, I had a lot more influence on the culture than I`
3. `act=30.25` target=`' influence'` snippet: `When we were a smaller organization, I had a lot more influence on the culture than`
4. `act=30.20` target=`' lot'` snippet: ` “When we were a smaller organization, I had a lot more influence on the`
5. `act=29.48` target=`' had'` snippet: ` More  “When we were a smaller organization, I had a lot more influence`
6. `act=29.04` target=`'aces'` snippet: ` a plebeian. His security guy is about 20 paces behind him. pic`
7. `act=28.54` target=`'dist'` snippet: `font-otj.lua share/texmf-dist/tex/l`
8. `act=28.53` target=`' CA'` snippet: ` sequencer (Applied Biosystems/Life Technologies, Carlsbad, CA, USA); the`
9. `act=28.49` target=`'11'` snippet: `0100 +++ contrib/virt.te	2012-11-25 21:`
10. `act=28.39` target=`'dist'` snippet: `-font-con.lua share/texmf-dist/tex/l`

### Feature #12117

Top 10 max-activating snippets:

1. `act=30.69` target=`'-'` snippet: `Sandhu *et al*, 2001](#bib19){ref-type="other"}),`
2. `act=30.57` target=`'-'` snippet: ` TRAIL resistant ([Figure 1d](#fig1){ref-type="fig"}).`
3. `act=30.22` target=`'.'` snippet: `DTP) independently of other extreme pressure additives. Further U.S. Pat.`
4. `act=29.85` target=`' respect'` snippet: `LR) moment functions for GMM, where the derivative with respect to first step non`
5. `act=29.76` target=`'.'` snippet: ` of other extreme pressure additives. Further U.S. Pat. No. 3,`
6. `act=29.72` target=`'ia'` snippet: `de 2012 a 2015) e analisa os impactos sociais das decisões`
7. `act=29.70` target=`'\\n'` snippet: `0012-00000008463-i03){#FIG3}  After discussion with`
8. `act=29.49` target=`'-'` snippet: `.[12](#jah32587-bib-0012){ref-type="ref"},`
9. `act=29.37` target=`'-'` snippet: `10](#mrm27594-bib-0010){ref-type="ref"}`
10. `act=29.23` target=`'-'` snippet: `32](#mrm27594-bib-0032){ref-type="ref"}`

### Feature #10047

Top 10 max-activating snippets:

1. `act=25.21` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
2. `act=25.21` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
3. `act=24.71` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
4. `act=24.70` target=`'\r\\n\t\t\t'` snippet: `>  			<xsl:element name="IntValue">  				<xsl:`
5. `act=24.69` target=`'\\n\t\t\t'` snippet: `function drawCurve(curve,steps,className) { 				if(!className`
6. `act=24.61` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
7. `act=24.60` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
8. `act=24.59` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=24.57` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
10. `act=24.52` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`

### Feature #7615

Top 10 max-activating snippets:

1. `act=25.11` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=25.03` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=24.97` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
4. `act=24.97` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=24.96` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=24.86` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
7. `act=24.85` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
8. `act=24.83` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
9. `act=24.82` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
10. `act=24.82` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`

### Feature #13227

Top 10 max-activating snippets:

1. `act=24.86` target=`'.'` snippet: `DTP) independently of other extreme pressure additives. Further U.S. Pat.`
2. `act=24.51` target=`'="'` snippet: ` (Table  [4](#Tab4){ref-type="table"}).Table 3`
3. `act=24.12` target=`')'` snippet: `:1983 in sports in Georgia (U.S. state)= -371 -`
4. `act=23.24` target=`'-'` snippet: `:20th-century French singers Category:20th-century male singersThis`
5. `act=23.02` target=`'oc'` snippet: ` to talk about military options, and he has done so unequivocally. Today he`
6. `act=23.00` target=`'-'` snippet: `Sandhu *et al*, 2001](#bib19){ref-type="other"}),`
7. `act=22.91` target=`'-'` snippet: ` and out-of-plane spin projection, respectively.[]{data-label="Fig1`
8. `act=22.83` target=`'.'` snippet: ` of other extreme pressure additives. Further U.S. Pat. No. 3,`
9. `act=22.68` target=`' OF'` snippet: ` "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either`
10. `act=22.65` target=`'-'` snippet: ` and ([2](#eq14){ref-type="disp-formula"}) are valid`

### Feature #13131

Top 10 max-activating snippets:

1. `act=23.80` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=23.70` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=23.62` target=`' \\n'` snippet: `inos (Campesinos an Workers Revolutionary Confederacy)   Category:National`
4. `act=23.55` target=`'\r\\n'` snippet: ` class="nav navbar-nav navbar-right">          <li><`
5. `act=23.52` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=23.51` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
7. `act=23.51` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
8. `act=23.44` target=`'\\n'` snippet: `:1948 births Category:Alumni of Imperial College London Category:Headmas`
9. `act=23.42` target=`'\\n'` snippet: ` favourable.  References  Category:Thai films Category:1958 films`
10. `act=23.41` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `

### Feature #10045

Top 10 max-activating snippets:

1. `act=23.38` target=`' mobil'` snippet: ` will have been made available by 2010.  Popular mobilisation needed  `
2. `act=22.01` target=`'ulose'` snippet: ` reported SIs produce varying proportions of the isomer trehalulose (1-O`
3. `act=21.87` target=`' cr'` snippet: `. So let’s hope the Japanese brand has managed to cram in plenty of`
4. `act=21.83` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
5. `act=21.78` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`
6. `act=21.73` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
7. `act=21.73` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
8. `act=21.67` target=`' hydrogen'` snippet: ` closed conformation of AdoHcyase are identified as the hydrogen bonds between the backbone`
9. `act=21.67` target=`' vitamin'` snippet: ` children aged 6 months to 5 years in Guatemala. Rates of vitamin and mineral deficiencies can`
10. `act=21.64` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`

### Feature #11488

Top 10 max-activating snippets:

1. `act=23.16` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=23.12` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=22.92` target=`'\\n'` snippet: `       Category:1953 births Category:Living people Category:Place of`
4. `act=22.90` target=`'.'` snippet: `Id}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vpn`
5. `act=22.88` target=`'\\n'` snippet: `home-watching-sun-move/  ====== jbrun `
6. `act=22.83` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
7. `act=22.83` target=`'.'` snippet: `, and we affirmed the trial court's judgment. State v.  McDoug`
8. `act=22.81` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
9. `act=22.78` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
10. `act=22.76` target=`'\\n'` snippet: `References  Category:1942 births Category:Living people Category:American ten`

### Feature #10925

Top 10 max-activating snippets:

1. `act=22.79` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
2. `act=22.58` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
3. `act=22.56` target=`'\\n\\n'` snippet: ` and closing rebuttal argument warrants reversal; (4) the   trial court imposed an`
4. `act=22.54` target=`'.'` snippet: ` the azimuthal direction *φ*(*r*) in [Fig. 2a](#f`
5. `act=22.53` target=`'.'` snippet: ` reticle for electron beam projection by a resist process (FIG. 3C). A`
6. `act=22.52` target=`'\\n\\n'` snippet: `}  and the result of the above code is as  m of 103,`
7. `act=22.52` target=`'.'` snippet: `MI Hamiltonian.[]{data-label="Fig2"}](Fig2.pdf){width="`
8. `act=22.52` target=`'\\n\\n'` snippet: `, “Trujillo contends that the items removed from    the home .`
9. `act=22.51` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
10. `act=22.50` target=`'.'` snippet: ` clear-cuts) in different parts of the property \[[@pone.0235320.`

### Feature #13821

Top 10 max-activating snippets:

1. `act=20.22` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=20.17` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=20.09` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
4. `act=20.06` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
5. `act=20.05` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
6. `act=19.94` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
7. `act=19.91` target=`'\r\\n\t'` snippet: `position: relative;  		z-index: 1;  		overflow: hidden`
8. `act=19.91` target=`'\\n\t\t'` snippet: ` in which the item lives (or will be added to) 		 * @param {`
9. `act=19.91` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
10. `act=19.90` target=`'\\n'` snippet: `;   enum {     Options = _Options,     Flags = traits`

### Feature #12697

Top 10 max-activating snippets:

1. `act=20.20` target=`' or'` snippet: `debug1: key_load_public: No such file or directory debug1`
2. `act=19.23` target=`'ury'` snippet: `Century Team DVD from Amazon.com All-Century Team Information from Baseball`
3. `act=18.52` target=`'-'` snippet: `off valves. Advantageously, the movable quick-coupling part spring-`
4. `act=18.38` target=`' or'` snippet: `debug1: key_load_public: No such file or directory debug1`
5. `act=18.16` target=`'Lower'` snippet: `      get { return System.Label.Weekly.toLowerCase().trim();`
6. `act=18.07` target=`'-'` snippet: ` a piston/cylinder arrangement. To lock the quick-coupling parts in the`
7. `act=17.95` target=`'-'` snippet: ` shift amount from the above zero-cross point of the U-phase inductive voltage 12`
8. `act=17.95` target=`'-'` snippet: ` pulse S1 and is counted down in response to the negative-direction position command pulse`
9. `act=17.93` target=`'-'` snippet: ` arranged fixedly on one structural component, while the other quick-coupling part is arranged`
10. `act=17.93` target=`'_'` snippet: ` 'rubocop'   s.add_development_dependency 'ruboc`

### Feature #15245

Top 10 max-activating snippets:

1. `act=18.78` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
2. `act=18.78` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
3. `act=18.73` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
4. `act=18.71` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
5. `act=18.62` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
6. `act=18.55` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
7. `act=18.52` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
8. `act=18.52` target=`'.'` snippet: ` *r* goes to *R*~c~, [Fig. 2a](#f`
9. `act=18.52` target=`'.'` snippet: ` the azimuthal direction *φ*(*r*) in [Fig. 2a](#f`
10. `act=18.50` target=`'.'` snippet: `Id}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vpn`

### Feature #15577

Top 10 max-activating snippets:

1. `act=17.41` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=17.25` target=`'.'` snippet: ` and 12 min after dipyridamole infusion (0.84 mg kg-`
3. `act=17.20` target=`'.'` snippet: `.googleapis.com/ajax/libs/jquery/1.12.4/`
4. `act=17.16` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
5. `act=17.13` target=`'\\n\\n'` snippet: `, “Trujillo contends that the items removed from    the home .`
6. `act=17.09` target=`'\\n\\n'` snippet: ` and closing rebuttal argument warrants reversal; (4) the   trial court imposed an`
7. `act=17.08` target=`'.'` snippet: ` 0%); fenugreek seed extract (FSE) 0.05%, basal diet`
8. `act=17.06` target=`'.'` snippet: `/vol) with phosphate buffer saline solution (PBS; 0.1M, pH`
9. `act=17.04` target=`'\\n\\n'` snippet: `}  and the result of the above code is as  m of 103,`
10. `act=17.02` target=`'.'` snippet: ` notably varied among individuals, with an average maximum consumption of 0.106 ± 0.`

### Feature #5196

Top 9 max-activating snippets:

1. `act=16.84` target=`' ----------------'` snippet: `istence.xml";     }      /* --------- UPDATE ---------------- */     @`
2. `act=14.92` target=`' ---------'` snippet: `/persistence.xml";     }      /* --------- UPDATE ---------------- */ `
3. `act=12.74` target=`' */'` snippet: `.xml";     }      /* --------- UPDATE ---------------- */     @Bean`
4. `act=10.14` target=`' UPDATE'` snippet: `persistence.xml";     }      /* --------- UPDATE ---------------- */     `
5. `act=0.75` target=`'^{'` snippet: `ray ($L_X = 2.5\times10^{39}erg\,`
6. `act=0.67` target=`'^{'` snippet: `1/e) \times 2^{1600} = 10^{481} \times`
7. `act=0.61` target=`'^{'` snippet: `7.5\pm0.5) \times 10^{39} erg\,`
8. `act=0.45` target=`' a'` snippet: ` testing. Results of the first MRI were inconclusive, but a second test confirmed the`
9. `act=0.34` target=`' our'` snippet: ` the truth of proposed moral norms, including those norms pertaining to our obligations concerning the natural`

### Feature #7448

Top 10 max-activating snippets:

1. `act=16.79` target=`'\\n\t\t\t'` snippet: ` { 					item[prop] = options[prop]; 				} 			`
2. `act=16.69` target=`'\\n\t\t\t'` snippet: `function drawCurve(curve,steps,className) { 				if(!className`
3. `act=16.69` target=`'\\n\t\t\t'` snippet: `, ok := err.(*errors.Validation); ok { 				return ve.`
4. `act=16.68` target=`'\\n\t'` snippet: `$("#id_permissions_role").sSelect(); 		$("#new`
5. `act=16.64` target=`'\\n\t'` snippet: `_poll: device-name freq iterations\n"); 		return; `
6. `act=16.63` target=`'\\n\t'` snippet: `_of_L1Directories" value="0"/> 		<param name`
7. `act=16.63` target=`'\\n\t'` snippet: ` style="width: 25%">25 %</div> 		</div>`
8. `act=16.62` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=16.62` target=`'\\n\t'` snippet: `', 'right', 'bottom', and 'chartArea' 	 * @prop {`
10. `act=16.60` target=`'\\n\t'` snippet: ` getPadding -  Returns an object with padding on the edges 	 * @prop {`

### Feature #14584

Top 10 max-activating snippets:

1. `act=15.47` target=`' hang'` snippet: ` lead after the first period at heavyweight but was unable to hang on.  `
2. `act=15.12` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=15.12` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=14.98` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
5. `act=14.89` target=`'it'` snippet: `onsumiert werden kann. Moles zielte damit auf das Potenz`
6. `act=14.89` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
7. `act=14.87` target=`'ipp'` snippet: ` the base of his penis. Brooke hogan hard nipples Big cocks`
8. `act=14.85` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=14.84` target=`'\\n\t\t'` snippet: ` in which the item lives (or will be added to) 		 * @param {`
10. `act=14.84` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`

### Feature #10545

Top 10 max-activating snippets:

1. `act=15.18` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
2. `act=15.08` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=15.08` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=14.96` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=14.86` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
6. `act=14.82` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`
7. `act=14.82` target=`'\\n\t\t'` snippet: ` Student").ExecuteUpdate(); 				session.Flush(); 			} 			`
8. `act=14.82` target=`'\\n\t\t'` snippet: ` top axis and 			// B1 is the bottom axis 			// There are`
9. `act=14.78` target=`'\\n\t\t\t'` snippet: ` { 					item[prop] = options[prop]; 				} 			`
10. `act=14.76` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`

### Feature #1989

Top 10 max-activating snippets:

1. `act=15.15` target=`'.'` snippet: ` heavy fines, in both the UK and USA: £1.8m for E`
2. `act=15.12` target=`'.'` snippet: `.googleapis.com/ajax/libs/jquery/1.12.4/`
3. `act=15.12` target=`'.'` snippet: `6]\]. According to the World Health Organisation (WHO), 1.7 -- 4%`
4. `act=15.12` target=`'\\n'` snippet: `  add your own caption  add your own caption  add your own`
5. `act=15.06` target=`'.'` snippet: ` loves and hates). She’s passionate about books,1. Field of the Invention`
6. `act=15.02` target=`'.'` snippet: ` og i perioder har han arbejdet i Senegal. Det sidste for`
7. `act=15.00` target=`'.'` snippet: ` and 12 min after dipyridamole infusion (0.84 mg kg-`
8. `act=14.99` target=`'.'` snippet: ` whom required conversion to laparotomy, with one mortality (1.22%). Fifty-`
9. `act=14.98` target=`'.'` snippet: `AXY Tab 7.0 Plus will come with Android 3.2 Honeycomb out`
10. `act=14.96` target=`'.'` snippet: ` of EMT-TFs was reported in 662 (49.6%) of the`

### Feature #15950

Top 10 max-activating snippets:

1. `act=15.14` target=`'fig'` snippet: `, E and F](#F1){ref-type="fig"}). In combination with`
2. `act=14.65` target=`'fig'` snippet: `Fig. 1A](#F1){ref-type="fig"} and fig.`
3. `act=14.59` target=`'fig'` snippet: ` ([Figure 2e](#fig2){ref-type="fig"} and [Supplementary`
4. `act=14.22` target=`'fig'` snippet: `Fig. 1B](#F1){ref-type="fig"}, upper). Moreover`
5. `act=14.20` target=`'fig'` snippet: ` [Fig. 1](#f1){ref-type="fig"}, *k*~*`
6. `act=14.18` target=`'fig'` snippet: `Fig. 2A](#F2){ref-type="fig"}, upper and middle`
7. `act=14.05` target=`'fig'` snippet: `Fig. 1a](#f1){ref-type="fig"} is osmotically`
8. `act=13.74` target=`' Cir'` snippet: ` State, 318 F.2d 852 (4th Cir.1963); Sl`
9. `act=13.71` target=`'fig'` snippet: ` ([Fig. 1](#fig0005){ref-type="fig"}B). Moreover`
10. `act=13.56` target=`'ref'` snippet: `rm27594-bib-0016){ref-type="ref"}, [17](#`

### Feature #2757

Top 10 max-activating snippets:

1. `act=14.92` target=`' \\n'` snippet: `inos (Campesinos an Workers Revolutionary Confederacy)   Category:National`
2. `act=14.87` target=`'\\n\t'` snippet: `MODE;				\ 	else								\ 		regs->ARM`
3. `act=14.85` target=`' \\n'` snippet: `.  Key achievements  References  External links   LifeArc's website`
4. `act=14.83` target=`' \\n'` snippet: `can we specify a partition column which is not a primary key   Yes, as`
5. `act=14.80` target=`'\\n\t'` snippet: `_of_L1Directories" value="0"/> 		<param name`
6. `act=14.77` target=`'\\n'` snippet: `regularly-how-hard-can-it-be ====== soph`
7. `act=14.77` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
8. `act=14.77` target=`' \\n'` snippet: ` and O_WRONLY) without conflicts with any others?   A: `
9. `act=14.77` target=`'\\n\t'` snippet: `_poll: device-name freq iterations\n"); 		return; `
10. `act=14.77` target=`'\\n\t'` snippet: `', 'right', 'bottom', and 'chartArea' 	 * @prop {`

### Feature #4781

Top 10 max-activating snippets:

1. `act=14.51` target=`'�'` snippet: `「国と県の裁判を見守る」と`
2. `act=1.11` target=`'ン�'` snippet: `、染谷有香だ。スレンダーなス`
3. `act=0.94` target=`'�'` snippet: `ಯ ಹಬ್ಬ ಹರಿ�`
4. `act=0.93` target=`'�'` snippet: `歌声を披露した巨乳ナース`
5. `act=0.80` target=`'�'` snippet: `受け、渡具知氏は報道陣に「名`
6. `act=0.79` target=`'�'` snippet: `精神で今年もファンを盛り上げ`
7. `act=0.59` target=`'-'` snippet: `uatex/luaotfload/fontloader-l-io.lua `
8. `act=0.58` target=`'τ'` snippet: `ταν στην Αθήνα κι αφότου πέρα`
9. `act=0.46` target=`'‐'` snippet: ` therapy (CRT), with CRT‐defibrillation (CRT‐D) or without`
10. `act=0.45` target=`'�'` snippet: `ロ写真」も好評。豊満なスタ`

### Feature #4170

Top 10 max-activating snippets:

1. `act=14.19` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=14.18` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=14.16` target=`'\\n\t\t'` snippet: `| C1 |                           | C2 |    | 			// |    `
4. `act=14.16` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
5. `act=14.15` target=`'\\n\t\t'` snippet: `<param name="decode_width" value="1"/> 			<!-- decode_`
6. `act=14.15` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
7. `act=14.12` target=`'\\n\t\t'` snippet: `param name="clock_rate" value="3500"/> 			<param name`
8. `act=14.11` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
9. `act=14.11` target=`'\\n\t\t'` snippet: `="pipeline_depth" value="6,6"/> 			<!-- pipeline depth`
10. `act=14.10` target=`'.'` snippet: ` salt, as fractionation of naked capsids in a 1.18-g/`

### Feature #16109

Top 5 max-activating snippets:

1. `act=13.98` target=`'�'` snippet: `��とひょうきんさで人々を楽`
2. `act=13.27` target=`'�'` snippet: `処女厨と呼ばれる人々からも注`
3. `act=9.31` target=`'�'` snippet: `度の高い国会議員を次々と応`
4. `act=0.97` target=`'ij'` snippet: ` promote the revolution of the persons and weeks, but criticxij for repairing the palp`
5. `act=0.44` target=`']{'` snippet: ` very much.  A:  「[一]{いち}を`


## Section B — Top 30 features by mean of top-K activations

These features have the most consistent strong activations across their top examples.
Often more 'reliable' interpretable features than Section A's peak-driven ones.

### Feature #121

Top 10 max-activating snippets:

1. `act=107.64` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
2. `act=102.72` target=`' Astr'` snippet: ` says research published in the journal Monthly Notices of the Royal Astronomical Society.`
3. `act=102.72` target=`' Astr'` snippet: `Category:2011 establishments in ScotlandGuest Blogger: A Christian Astronomer Reflect`
4. `act=102.15` target=`' cr'` snippet: `. So let’s hope the Japanese brand has managed to cram in plenty of`
5. `act=102.07` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
6. `act=101.94` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
7. `act=101.88` target=`'agles'` snippet: ` Bruce Arians was a candidate for the vacant Philadelphia Eagles job, but the`
8. `act=101.78` target=`' cave'` snippet: `feedback, Dickerson said.  Adcock said one caveat of the study`
9. `act=100.88` target=`' Borough'` snippet: ` to you by the Colchester Travel Plan Club, Colchester Borough Council Air Quality Team`
10. `act=100.80` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`

### Feature #6630

Top 10 max-activating snippets:

1. `act=52.59` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`
2. `act=52.40` target=`' initiated'` snippet: `pson, Swope said at the time the study was initiated.  The`
3. `act=51.91` target=`' mobil'` snippet: ` will have been made available by 2010.  Popular mobilisation needed  `
4. `act=51.68` target=`' hydrogen'` snippet: ` closed conformation of AdoHcyase are identified as the hydrogen bonds between the backbone`
5. `act=51.58` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
6. `act=51.57` target=`' caspase'` snippet: ` programmed cell death pathway of the cells from patient group A was caspase-3 and poly`
7. `act=51.41` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
8. `act=51.37` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
9. `act=51.25` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
10. `act=51.10` target=`' lign'` snippet: `intéressé n'a pu être joint. La ligne télé`

### Feature #2255

Top 10 max-activating snippets:

1. `act=44.43` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=44.22` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=44.15` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
4. `act=44.15` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
5. `act=43.98` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
6. `act=43.86` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
7. `act=43.85` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
8. `act=43.79` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
9. `act=43.75` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
10. `act=43.71` target=`'\\n'` snippet: `:1948 births Category:Alumni of Imperial College London Category:Headmas`

### Feature #12520

Top 10 max-activating snippets:

1. `act=36.48` target=`' initiated'` snippet: `pson, Swope said at the time the study was initiated.  The`
2. `act=36.10` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
3. `act=36.02` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`
4. `act=35.99` target=`' dismissed'` snippet: ` record on appeal. The State argues defendants' appeal must be dismissed because the trial court`
5. `act=35.96` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
6. `act=35.96` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
7. `act=35.91` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
8. `act=35.87` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
9. `act=35.86` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
10. `act=35.57` target=`'ichi'` snippet: ` Ishizuka joined team ALIVE which is based in Aichi prefecture. `

### Feature #6767

Top 10 max-activating snippets:

1. `act=33.51` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=33.24` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=33.22` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
4. `act=33.22` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
5. `act=33.08` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=33.03` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
7. `act=32.98` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
8. `act=32.97` target=`'\\n\t'` snippet: ` getPadding -  Returns an object with padding on the edges 	 * @prop {`
9. `act=32.97` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
10. `act=32.96` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`

### Feature #6484

Top 10 max-activating snippets:

1. `act=32.50` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=32.48` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=32.48` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=32.21` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=32.14` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
6. `act=32.00` target=`'\r\\n\t\t\t'` snippet: `>  			<xsl:element name="IntValue">  				<xsl:`
7. `act=31.96` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
8. `act=31.94` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
9. `act=31.81` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
10. `act=31.80` target=`'\\n\t\t\t'` snippet: ` -1 			if result.Response() != nil { 				sc = result`

### Feature #15230

Top 10 max-activating snippets:

1. `act=32.11` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=32.08` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=31.92` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
4. `act=31.88` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
5. `act=31.75` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
6. `act=31.74` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
7. `act=31.74` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
8. `act=31.73` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
9. `act=31.71` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
10. `act=31.68` target=`'\\n'` snippet: `regularly-how-hard-can-it-be ====== soph`

### Feature #12117

Top 10 max-activating snippets:

1. `act=30.69` target=`'-'` snippet: `Sandhu *et al*, 2001](#bib19){ref-type="other"}),`
2. `act=30.57` target=`'-'` snippet: ` TRAIL resistant ([Figure 1d](#fig1){ref-type="fig"}).`
3. `act=30.22` target=`'.'` snippet: `DTP) independently of other extreme pressure additives. Further U.S. Pat.`
4. `act=29.85` target=`' respect'` snippet: `LR) moment functions for GMM, where the derivative with respect to first step non`
5. `act=29.76` target=`'.'` snippet: ` of other extreme pressure additives. Further U.S. Pat. No. 3,`
6. `act=29.72` target=`'ia'` snippet: `de 2012 a 2015) e analisa os impactos sociais das decisões`
7. `act=29.70` target=`'\\n'` snippet: `0012-00000008463-i03){#FIG3}  After discussion with`
8. `act=29.49` target=`'-'` snippet: `.[12](#jah32587-bib-0012){ref-type="ref"},`
9. `act=29.37` target=`'-'` snippet: `10](#mrm27594-bib-0010){ref-type="ref"}`
10. `act=29.23` target=`'-'` snippet: `32](#mrm27594-bib-0032){ref-type="ref"}`

### Feature #5747

Top 10 max-activating snippets:

1. `act=30.75` target=`' faced'` snippet: ` a combination of both?Those are just some of the questions faced by today’s`
2. `act=30.44` target=`' on'` snippet: ` we were a smaller organization, I had a lot more influence on the culture than I`
3. `act=30.25` target=`' influence'` snippet: `When we were a smaller organization, I had a lot more influence on the culture than`
4. `act=30.20` target=`' lot'` snippet: ` “When we were a smaller organization, I had a lot more influence on the`
5. `act=29.48` target=`' had'` snippet: ` More  “When we were a smaller organization, I had a lot more influence`
6. `act=29.04` target=`'aces'` snippet: ` a plebeian. His security guy is about 20 paces behind him. pic`
7. `act=28.54` target=`'dist'` snippet: `font-otj.lua share/texmf-dist/tex/l`
8. `act=28.53` target=`' CA'` snippet: ` sequencer (Applied Biosystems/Life Technologies, Carlsbad, CA, USA); the`
9. `act=28.49` target=`'11'` snippet: `0100 +++ contrib/virt.te	2012-11-25 21:`
10. `act=28.39` target=`'dist'` snippet: `-font-con.lua share/texmf-dist/tex/l`

### Feature #7615

Top 10 max-activating snippets:

1. `act=25.11` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=25.03` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=24.97` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
4. `act=24.97` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=24.96` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=24.86` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
7. `act=24.85` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
8. `act=24.83` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
9. `act=24.82` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
10. `act=24.82` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`

### Feature #10047

Top 10 max-activating snippets:

1. `act=25.21` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
2. `act=25.21` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
3. `act=24.71` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
4. `act=24.70` target=`'\r\\n\t\t\t'` snippet: `>  			<xsl:element name="IntValue">  				<xsl:`
5. `act=24.69` target=`'\\n\t\t\t'` snippet: `function drawCurve(curve,steps,className) { 				if(!className`
6. `act=24.61` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
7. `act=24.60` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
8. `act=24.59` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=24.57` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
10. `act=24.52` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`

### Feature #13131

Top 10 max-activating snippets:

1. `act=23.80` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=23.70` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=23.62` target=`' \\n'` snippet: `inos (Campesinos an Workers Revolutionary Confederacy)   Category:National`
4. `act=23.55` target=`'\r\\n'` snippet: ` class="nav navbar-nav navbar-right">          <li><`
5. `act=23.52` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
6. `act=23.51` target=`'\\n'` snippet: ` ships Category:Age of Sail merchant ships of England Category:Sealing`
7. `act=23.51` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
8. `act=23.44` target=`'\\n'` snippet: `:1948 births Category:Alumni of Imperial College London Category:Headmas`
9. `act=23.42` target=`'\\n'` snippet: ` favourable.  References  Category:Thai films Category:1958 films`
10. `act=23.41` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `

### Feature #13227

Top 10 max-activating snippets:

1. `act=24.86` target=`'.'` snippet: `DTP) independently of other extreme pressure additives. Further U.S. Pat.`
2. `act=24.51` target=`'="'` snippet: ` (Table  [4](#Tab4){ref-type="table"}).Table 3`
3. `act=24.12` target=`')'` snippet: `:1983 in sports in Georgia (U.S. state)= -371 -`
4. `act=23.24` target=`'-'` snippet: `:20th-century French singers Category:20th-century male singersThis`
5. `act=23.02` target=`'oc'` snippet: ` to talk about military options, and he has done so unequivocally. Today he`
6. `act=23.00` target=`'-'` snippet: `Sandhu *et al*, 2001](#bib19){ref-type="other"}),`
7. `act=22.91` target=`'-'` snippet: ` and out-of-plane spin projection, respectively.[]{data-label="Fig1`
8. `act=22.83` target=`'.'` snippet: ` of other extreme pressure additives. Further U.S. Pat. No. 3,`
9. `act=22.68` target=`' OF'` snippet: ` "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either`
10. `act=22.65` target=`'-'` snippet: ` and ([2](#eq14){ref-type="disp-formula"}) are valid`

### Feature #11488

Top 10 max-activating snippets:

1. `act=23.16` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=23.12` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=22.92` target=`'\\n'` snippet: `       Category:1953 births Category:Living people Category:Place of`
4. `act=22.90` target=`'.'` snippet: `Id}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vpn`
5. `act=22.88` target=`'\\n'` snippet: `home-watching-sun-move/  ====== jbrun `
6. `act=22.83` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
7. `act=22.83` target=`'.'` snippet: `, and we affirmed the trial court's judgment. State v.  McDoug`
8. `act=22.81` target=`'\\n'` snippet: ` - Friday plus breaking news alerts, by email Update newsletter preferences  Donald Trump has`
9. `act=22.78` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
10. `act=22.76` target=`'\\n'` snippet: `References  Category:1942 births Category:Living people Category:American ten`

### Feature #10925

Top 10 max-activating snippets:

1. `act=22.79` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
2. `act=22.58` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
3. `act=22.56` target=`'\\n\\n'` snippet: ` and closing rebuttal argument warrants reversal; (4) the   trial court imposed an`
4. `act=22.54` target=`'.'` snippet: ` the azimuthal direction *φ*(*r*) in [Fig. 2a](#f`
5. `act=22.53` target=`'.'` snippet: ` reticle for electron beam projection by a resist process (FIG. 3C). A`
6. `act=22.52` target=`'\\n\\n'` snippet: `}  and the result of the above code is as  m of 103,`
7. `act=22.52` target=`'.'` snippet: `MI Hamiltonian.[]{data-label="Fig2"}](Fig2.pdf){width="`
8. `act=22.52` target=`'\\n\\n'` snippet: `, “Trujillo contends that the items removed from    the home .`
9. `act=22.51` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
10. `act=22.50` target=`'.'` snippet: ` clear-cuts) in different parts of the property \[[@pone.0235320.`

### Feature #10045

Top 10 max-activating snippets:

1. `act=23.38` target=`' mobil'` snippet: ` will have been made available by 2010.  Popular mobilisation needed  `
2. `act=22.01` target=`'ulose'` snippet: ` reported SIs produce varying proportions of the isomer trehalulose (1-O`
3. `act=21.87` target=`' cr'` snippet: `. So let’s hope the Japanese brand has managed to cram in plenty of`
4. `act=21.83` target=`'rons'` snippet: ` Corps Air Station Miramar that is currently composed of five squadrons and one battalion that`
5. `act=21.78` target=`' NAD'` snippet: `/inhibitor binding, not the oxidation state of the bound NAD. The adenine`
6. `act=21.73` target=`' rue'` snippet: `aux » qu'il occupe sur son domaine, rue de la Boul`
7. `act=21.73` target=`'lor'` snippet: `opyranoside: Possible mechanisms for Polygoni Multiflori Radix-`
8. `act=21.67` target=`' hydrogen'` snippet: ` closed conformation of AdoHcyase are identified as the hydrogen bonds between the backbone`
9. `act=21.67` target=`' vitamin'` snippet: ` children aged 6 months to 5 years in Guatemala. Rates of vitamin and mineral deficiencies can`
10. `act=21.64` target=`'vy'` snippet: `' title: Asymptotic equivalence for pure jump Lévy processes with unknown Lé`

### Feature #13821

Top 10 max-activating snippets:

1. `act=20.22` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=20.17` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=20.09` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
4. `act=20.06` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
5. `act=20.05` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
6. `act=19.94` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
7. `act=19.91` target=`'\r\\n\t'` snippet: `position: relative;  		z-index: 1;  		overflow: hidden`
8. `act=19.91` target=`'\\n\t\t'` snippet: ` in which the item lives (or will be added to) 		 * @param {`
9. `act=19.91` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
10. `act=19.90` target=`'\\n'` snippet: `;   enum {     Options = _Options,     Flags = traits`

### Feature #15245

Top 10 max-activating snippets:

1. `act=18.78` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
2. `act=18.78` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
3. `act=18.73` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
4. `act=18.71` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
5. `act=18.62` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
6. `act=18.55` target=`'.'` snippet: ` as gp130/JAK-STAT3 signaling ([Fig. 2](#fig0010`
7. `act=18.52` target=`'\\n'` snippet: ` ContentApi(             show_archived=True,             show_de`
8. `act=18.52` target=`'.'` snippet: ` *r* goes to *R*~c~, [Fig. 2a](#f`
9. `act=18.52` target=`'.'` snippet: ` the azimuthal direction *φ*(*r*) in [Fig. 2a](#f`
10. `act=18.50` target=`'.'` snippet: `Id}/resourceGroups/{resourceGroupName}/providers/Microsoft.Network/vpn`

### Feature #12697

Top 10 max-activating snippets:

1. `act=20.20` target=`' or'` snippet: `debug1: key_load_public: No such file or directory debug1`
2. `act=19.23` target=`'ury'` snippet: `Century Team DVD from Amazon.com All-Century Team Information from Baseball`
3. `act=18.52` target=`'-'` snippet: `off valves. Advantageously, the movable quick-coupling part spring-`
4. `act=18.38` target=`' or'` snippet: `debug1: key_load_public: No such file or directory debug1`
5. `act=18.16` target=`'Lower'` snippet: `      get { return System.Label.Weekly.toLowerCase().trim();`
6. `act=18.07` target=`'-'` snippet: ` a piston/cylinder arrangement. To lock the quick-coupling parts in the`
7. `act=17.95` target=`'-'` snippet: ` shift amount from the above zero-cross point of the U-phase inductive voltage 12`
8. `act=17.95` target=`'-'` snippet: ` pulse S1 and is counted down in response to the negative-direction position command pulse`
9. `act=17.93` target=`'-'` snippet: ` arranged fixedly on one structural component, while the other quick-coupling part is arranged`
10. `act=17.93` target=`'_'` snippet: ` 'rubocop'   s.add_development_dependency 'ruboc`

### Feature #15577

Top 10 max-activating snippets:

1. `act=17.41` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=17.25` target=`'.'` snippet: ` and 12 min after dipyridamole infusion (0.84 mg kg-`
3. `act=17.20` target=`'.'` snippet: `.googleapis.com/ajax/libs/jquery/1.12.4/`
4. `act=17.16` target=`'\\n\\n'` snippet: ` who has title to real property is still the owner of the   property even if there`
5. `act=17.13` target=`'\\n\\n'` snippet: `, “Trujillo contends that the items removed from    the home .`
6. `act=17.09` target=`'\\n\\n'` snippet: ` and closing rebuttal argument warrants reversal; (4) the   trial court imposed an`
7. `act=17.08` target=`'.'` snippet: ` 0%); fenugreek seed extract (FSE) 0.05%, basal diet`
8. `act=17.06` target=`'.'` snippet: `/vol) with phosphate buffer saline solution (PBS; 0.1M, pH`
9. `act=17.04` target=`'\\n\\n'` snippet: `}  and the result of the above code is as  m of 103,`
10. `act=17.02` target=`'.'` snippet: ` notably varied among individuals, with an average maximum consumption of 0.106 ± 0.`

### Feature #7448

Top 10 max-activating snippets:

1. `act=16.79` target=`'\\n\t\t\t'` snippet: ` { 					item[prop] = options[prop]; 				} 			`
2. `act=16.69` target=`'\\n\t\t\t'` snippet: `function drawCurve(curve,steps,className) { 				if(!className`
3. `act=16.69` target=`'\\n\t\t\t'` snippet: `, ok := err.(*errors.Validation); ok { 				return ve.`
4. `act=16.68` target=`'\\n\t'` snippet: `$("#id_permissions_role").sSelect(); 		$("#new`
5. `act=16.64` target=`'\\n\t'` snippet: `_poll: device-name freq iterations\n"); 		return; `
6. `act=16.63` target=`'\\n\t'` snippet: `_of_L1Directories" value="0"/> 		<param name`
7. `act=16.63` target=`'\\n\t'` snippet: ` style="width: 25%">25 %</div> 		</div>`
8. `act=16.62` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=16.62` target=`'\\n\t'` snippet: `', 'right', 'bottom', and 'chartArea' 	 * @prop {`
10. `act=16.60` target=`'\\n\t'` snippet: ` getPadding -  Returns an object with padding on the edges 	 * @prop {`

### Feature #1989

Top 10 max-activating snippets:

1. `act=15.15` target=`'.'` snippet: ` heavy fines, in both the UK and USA: £1.8m for E`
2. `act=15.12` target=`'.'` snippet: `.googleapis.com/ajax/libs/jquery/1.12.4/`
3. `act=15.12` target=`'.'` snippet: `6]\]. According to the World Health Organisation (WHO), 1.7 -- 4%`
4. `act=15.12` target=`'\\n'` snippet: `  add your own caption  add your own caption  add your own`
5. `act=15.06` target=`'.'` snippet: ` loves and hates). She’s passionate about books,1. Field of the Invention`
6. `act=15.02` target=`'.'` snippet: ` og i perioder har han arbejdet i Senegal. Det sidste for`
7. `act=15.00` target=`'.'` snippet: ` and 12 min after dipyridamole infusion (0.84 mg kg-`
8. `act=14.99` target=`'.'` snippet: ` whom required conversion to laparotomy, with one mortality (1.22%). Fifty-`
9. `act=14.98` target=`'.'` snippet: `AXY Tab 7.0 Plus will come with Android 3.2 Honeycomb out`
10. `act=14.96` target=`'.'` snippet: ` of EMT-TFs was reported in 662 (49.6%) of the`

### Feature #14584

Top 10 max-activating snippets:

1. `act=15.47` target=`' hang'` snippet: ` lead after the first period at heavyweight but was unable to hang on.  `
2. `act=15.12` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=15.12` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=14.98` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
5. `act=14.89` target=`'it'` snippet: `onsumiert werden kann. Moles zielte damit auf das Potenz`
6. `act=14.89` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`
7. `act=14.87` target=`'ipp'` snippet: ` the base of his penis. Brooke hogan hard nipples Big cocks`
8. `act=14.85` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
9. `act=14.84` target=`'\\n\t\t'` snippet: ` in which the item lives (or will be added to) 		 * @param {`
10. `act=14.84` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`

### Feature #10545

Top 10 max-activating snippets:

1. `act=15.18` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
2. `act=15.08` target=`'\r\\n\t\t'` snippet: `}  		}    		&.fade-in {  			&:before`
3. `act=15.08` target=`'\r\\n\t\t'` snippet: `">  		<xsl:element name="ObjectValue">  			<xsl:`
4. `act=14.96` target=`'\r\\n\t\t\t'` snippet: `  				&:before {  					opacity: 1;  				}  			`
5. `act=14.86` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
6. `act=14.82` target=`'\\n\t\t'` snippet: `	overflow: hidden; 				background: #000; 			}  			`
7. `act=14.82` target=`'\\n\t\t'` snippet: ` Student").ExecuteUpdate(); 				session.Flush(); 			} 			`
8. `act=14.82` target=`'\\n\t\t'` snippet: ` top axis and 			// B1 is the bottom axis 			// There are`
9. `act=14.78` target=`'\\n\t\t\t'` snippet: ` { 					item[prop] = options[prop]; 				} 			`
10. `act=14.76` target=`'\\n\t\t'` snippet: ` * @param {Chart} chart - the chart to use 		 * @param {`

### Feature #2757

Top 10 max-activating snippets:

1. `act=14.92` target=`' \\n'` snippet: `inos (Campesinos an Workers Revolutionary Confederacy)   Category:National`
2. `act=14.87` target=`'\\n\t'` snippet: `MODE;				\ 	else								\ 		regs->ARM`
3. `act=14.85` target=`' \\n'` snippet: `.  Key achievements  References  External links   LifeArc's website`
4. `act=14.83` target=`' \\n'` snippet: `can we specify a partition column which is not a primary key   Yes, as`
5. `act=14.80` target=`'\\n\t'` snippet: `_of_L1Directories" value="0"/> 		<param name`
6. `act=14.77` target=`'\\n'` snippet: `regularly-how-hard-can-it-be ====== soph`
7. `act=14.77` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`
8. `act=14.77` target=`' \\n'` snippet: ` and O_WRONLY) without conflicts with any others?   A: `
9. `act=14.77` target=`'\\n\t'` snippet: `_poll: device-name freq iterations\n"); 		return; `
10. `act=14.77` target=`'\\n\t'` snippet: `', 'right', 'bottom', and 'chartArea' 	 * @prop {`

### Feature #15950

Top 10 max-activating snippets:

1. `act=15.14` target=`'fig'` snippet: `, E and F](#F1){ref-type="fig"}). In combination with`
2. `act=14.65` target=`'fig'` snippet: `Fig. 1A](#F1){ref-type="fig"} and fig.`
3. `act=14.59` target=`'fig'` snippet: ` ([Figure 2e](#fig2){ref-type="fig"} and [Supplementary`
4. `act=14.22` target=`'fig'` snippet: `Fig. 1B](#F1){ref-type="fig"}, upper). Moreover`
5. `act=14.20` target=`'fig'` snippet: ` [Fig. 1](#f1){ref-type="fig"}, *k*~*`
6. `act=14.18` target=`'fig'` snippet: `Fig. 2A](#F2){ref-type="fig"}, upper and middle`
7. `act=14.05` target=`'fig'` snippet: `Fig. 1a](#f1){ref-type="fig"} is osmotically`
8. `act=13.74` target=`' Cir'` snippet: ` State, 318 F.2d 852 (4th Cir.1963); Sl`
9. `act=13.71` target=`'fig'` snippet: ` ([Fig. 1](#fig0005){ref-type="fig"}B). Moreover`
10. `act=13.56` target=`'ref'` snippet: `rm27594-bib-0016){ref-type="ref"}, [17](#`

### Feature #4170

Top 10 max-activating snippets:

1. `act=14.19` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=14.18` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
3. `act=14.16` target=`'\\n\t\t'` snippet: `| C1 |                           | C2 |    | 			// |    `
4. `act=14.16` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
5. `act=14.15` target=`'\\n\t\t'` snippet: `<param name="decode_width" value="1"/> 			<!-- decode_`
6. `act=14.15` target=`'\\n\\n'` snippet: ` except not sabotaged by trends and a karma system  Q:  `
7. `act=14.12` target=`'\\n\t\t'` snippet: `param name="clock_rate" value="3500"/> 			<param name`
8. `act=14.11` target=`'\\n\t\t'` snippet: `"/>		 			<!-- buffer between IF and ID stage --> 			<param name`
9. `act=14.11` target=`'\\n\t\t'` snippet: `="pipeline_depth" value="6,6"/> 			<!-- pipeline depth`
10. `act=14.10` target=`'.'` snippet: ` salt, as fractionation of naked capsids in a 1.18-g/`

### Feature #15247

Top 10 max-activating snippets:

1. `act=13.58` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
2. `act=13.47` target=`'\\n'` snippet: `English rugby union players Category:England international rugby union players Category:Barbar`
3. `act=13.42` target=`'\\n'` snippet: `air,           individually          and     as     personal  representative of`
4. `act=13.38` target=`'.'` snippet: `ugleich kenntnisreich“ hielt. Im Nachwort zu`
5. `act=13.37` target=`'.'` snippet: `.googleapis.com/ajax/libs/jquery/1.12.4/`
6. `act=13.36` target=`'.'` snippet: ` and a bar made of egg whites and crickets or something. But I knew this`
7. `act=13.35` target=`'\\n'` snippet: ` more essential elements of a claim or defense on which an  adverse party would`
8. `act=13.34` target=`'.'` snippet: `umber, dried mushrooms (a lot of 'em!), etc.  While I`
9. `act=13.34` target=`'\\n'` snippet: `hinney, his project manager at Andersen, gave  him a memorandum that`
10. `act=13.33` target=`'.'` snippet: `, Murchison, Cumming & Baker, Michael B. Lawler, Tob`

### Feature #3509

Top 10 max-activating snippets:

1. `act=13.39` target=`'\\n\t\t\\n\t'` snippet: `i-1]-l[i-1]+1; 		 		long long t`
2. `act=13.37` target=`'\\n\t'` snippet: `_of_L1Directories" value="0"/> 		<param name`
3. `act=13.35` target=`'\\n\t\t'` snippet: `fp_instruction_window_size" value="16"/> 			<!-- the instruction`
4. `act=13.31` target=`'\\n\t'` snippet: `="width: 5%">Something goes wrong</div> 		</div>`
5. `act=13.30` target=`'\\n\t\t'` snippet: ` phy_reg number is needed for renaming logic,  			renaming logic`
6. `act=13.30` target=`'\\n\t\t'` snippet: `<param name="decode_width" value="1"/> 			<!-- decode_`
7. `act=13.29` target=`'\\n\t'` snippet: `$("#id_permissions_role").sSelect(); 		$("#new`
8. `act=13.27` target=`'\\n\t\t'` snippet: `="pipeline_depth" value="6,6"/> 			<!-- pipeline depth`
9. `act=13.26` target=`'\\n'` snippet: `šov players Category:AS Trenčín players Category:MEAP`
10. `act=13.23` target=`'\\n\t'` snippet: `_poll: device-name freq iterations\n"); 		return; `

### Feature #411

Top 10 max-activating snippets:

1. `act=13.04` target=`'�'` snippet: `な存在といえば、篠崎愛だろ`
2. `act=12.77` target=`'�'` snippet: `没编译过的词典源码  4`
3. `act=12.69` target=`'ό'` snippet: `�ταν στην Αθήνα κι αφότου πέ`
4. `act=12.44` target=`'�'` snippet: `view class="weui-label">跳转链接</`
5. `act=12.34` target=`'氏'` snippet: `８９票、稲嶺氏が１万`
6. `act=12.24` target=`'�'` snippet: `�めてきたと主張。「移設を受`
7. `act=12.17` target=`'ύ'` snippet: ` με τον Ουασίμ Μπούι. Ο`
8. `act=12.12` target=`'月'` snippet: `oconuts Jakarta 发布于 2017年6月14日  `
9. `act=12.04` target=`'�'` snippet: `, with 200 ng/day\          6-week ♀ BALB/c`
10. `act=11.88` target=`'ά'` snippet: `ός, έχω λοιπόν μεγάλο κίνη`


## Section C — Random sample of 20 alive features (sanity check)

Random sample — typical features, not curated. Use this to estimate the prevalence
of monosemantic vs polysemantic features in the SAE overall.

### Feature #10515

Top 10 max-activating snippets:

1. `act=9.25` target=`' past'` snippet: ` do so by “ordering” them through a menu. Some pastas are better than`
2. `act=9.25` target=`' past'` snippet: ` at each other until one dies. You get credits for other pastas your own pasta`
3. `act=9.11` target=`' past'` snippet: ` people.  Do you know why the edges of the pasty are so thick`
4. `act=9.06` target=`' past'` snippet: `? Check this out. Remember, the secret to a great pasty is the crust`
5. `act=8.99` target=`' past'` snippet: ` them for lunch. By the mid-1800s, the pasty had been introduced`
6. `act=8.96` target=`' past'` snippet: `, pigeon, or wild game that was hunted. The pasty originated in England`
7. `act=8.94` target=`' past'` snippet: ` restaurant specializes in pasties.  Want to make a pasty? Check this`
8. `act=8.78` target=`' past'` snippet: ` a number of credits).  Once spawned, your pastas start flying around`
9. `act=8.50` target=`' past'` snippet: ` or cycling classes. Dr. Asin’s #1 pastime is spending time`
10. `act=8.44` target=`' past'` snippet: ` of razors.  3. What are your favorite pastimes and/or`

### Feature #1835

Top 10 max-activating snippets:

1. `act=4.53` target=`' food'` snippet: ` for child and household contextual variables. In lagged models, food insecurity was predictive of`
2. `act=4.50` target=`' food'` snippet: `. Inside, there's a small Village Grocer and multiple food establishments (for some`
3. `act=4.49` target=`' food'` snippet: `, the Chronicle notes that acquisitions and mergers in the Bay Area food world have been on`
4. `act=4.39` target=`' food'` snippet: ` thinking.  I’m talking about one of the food staples of the`
5. `act=4.35` target=`' food'` snippet: ` upper compartment. It's great because there's no way for food to get into the`
6. `act=4.19` target=`' food'` snippet: ` as diverse as Medicare and Medicaid, old age Social Security, food stamps, disability insurance`
7. `act=4.16` target=`' food'` snippet: ` at tax advisory firm Grant Thornton in San Francisco who works with food companies, told the`
8. `act=4.14` target=`' food'` snippet: `ovac camps were horrendous. Prisoners received minimal food. Shelter and`
9. `act=4.12` target=`' food'` snippet: `, most glorious..."  1) Do you have a food processor? Can you`
10. `act=4.09` target=`' food'` snippet: ` would have been used mainly between the spring and fall, when food would have been pl`

### Feature #411

Top 10 max-activating snippets:

1. `act=13.04` target=`'�'` snippet: `な存在といえば、篠崎愛だろ`
2. `act=12.77` target=`'�'` snippet: `没编译过的词典源码  4`
3. `act=12.69` target=`'ό'` snippet: `�ταν στην Αθήνα κι αφότου πέ`
4. `act=12.44` target=`'�'` snippet: `view class="weui-label">跳转链接</`
5. `act=12.34` target=`'氏'` snippet: `８９票、稲嶺氏が１万`
6. `act=12.24` target=`'�'` snippet: `�めてきたと主張。「移設を受`
7. `act=12.17` target=`'ύ'` snippet: ` με τον Ουασίμ Μπούι. Ο`
8. `act=12.12` target=`'月'` snippet: `oconuts Jakarta 发布于 2017年6月14日  `
9. `act=12.04` target=`'�'` snippet: `, with 200 ng/day\          6-week ♀ BALB/c`
10. `act=11.88` target=`'ά'` snippet: `ός, έχω λοιπόν μεγάλο κίνη`

### Feature #12195

Top 9 max-activating snippets:

1. `act=8.82` target=`'�'` snippet: ` Towns Liushan, Guangxi (流山), in Li`
2. `act=8.77` target=`'�'` snippet: ` Guangxi Liushan, Henan (留山), in N`
3. `act=8.05` target=`'�'` snippet: `an Liushan, Shandong (柳山), in Lin`
4. `act=8.00` target=`'�'` snippet: ` China:  Liushan Subdistrict (刘山街道`
5. `act=6.02` target=`'�'` snippet: `に入り「基地は経済発展の邪`
6. `act=5.65` target=`'�'` snippet: `えてくれ、明るい街に発展させて`
7. `act=0.71` target=`'大'` snippet: `由、社民、沖縄社会大衆推`
8. `act=0.54` target=`'大'` snippet: `�い、175センチという高身長で大人びた�`
9. `act=0.40` target=`' important'` snippet: ` Charters  As a family owned business we know how important it is that your`

### Feature #4527

Top 10 max-activating snippets:

1. `act=6.13` target=`' enter'` snippet: ` confidence in my faith. I’m always hoping i can enter into a very focused`
2. `act=6.13` target=`' enter'` snippet: ` church leading to the sacristy. They then tried to enter but were unable to`
3. `act=6.05` target=`' enter'` snippet: ` human beings. a ritual is not a ticket allowing one to enter a certain door or`
4. `act=6.01` target=`' entered'` snippet: ` gender revolution has been a one-sided effort. Women have entered previously male precinct`
5. `act=5.96` target=`' enter'` snippet: ` understanding and satisfactory knowledge on the subject that you are going to enter. If you’`
6. `act=5.92` target=`' entered'` snippet: ` as the perceiver of music may no longer exist, having entered into a unitive`
7. `act=5.86` target=`' entered'` snippet: `.  According to Sessler, Gordon said he entered a rehab facility`
8. `act=5.86` target=`' entered'` snippet: ` the ceremony master announced as the Conqueror and the Queen entered the Great Hall.`
9. `act=5.85` target=`' enter'` snippet: `" to a limited company allowed the organisation to employ staff, enter into contracts for accommodation`
10. `act=5.82` target=`' entering'` snippet: ` a former MI6 chief, told BBC that the world was entering an era possibly “`

### Feature #4028

Top 10 max-activating snippets:

1. `act=5.49` target=`'\\n'` snippet: `} \frac r2 & \frac r2\\ \frac{r`
2. `act=5.44` target=`'\\n'` snippet: `array}{cc} \cos\theta & 0\\ \sin\theta`
3. `act=5.07` target=`'\\'` snippet: ` \frac r2 & \frac r2\\ \frac{r}{`
4. `act=5.02` target=`' &'` snippet: `\begin{array}{cc} \frac r2 & \frac r2`
5. `act=4.74` target=`' &'` snippet: ` r2\\ \frac{r}{2d} &-\frac{r`
6. `act=4.72` target=`' &'` snippet: `\begin{array}{cc} \cos\theta & 0\\ \`
7. `act=4.11` target=`'\\'` snippet: `}{cc} \cos\theta & 0\\ \sin\theta &`
8. `act=4.08` target=`' &'` snippet: `\cos\theta & 0\\ \sin\theta & 0\\ 0`
9. `act=4.07` target=`'\\n'` snippet: `begin{array}{c} \omega_1\\ \omega_2`
10. `act=3.96` target=`'\\n'` snippet: ` = \left( \begin{array}{cc} \frac r2`

### Feature #3673

Top 10 max-activating snippets:

1. `act=5.15` target=`' $'` snippet: ` n)$, where $m$ is the number of features and $n$ is the`
2. `act=4.97` target=`'m'` snippet: `{O}(m^2 \cdot n)$, where $m$ is the number`
3. `act=4.73` target=`' $'` snippet: `(p_i)dp_i $$ where $p_i$`
4. `act=4.61` target=`'where'` snippet: `theta = d_0\dot\theta $$ where $d_0`
5. `act=4.61` target=`' $$\\'` snippet: `,, \qquad \alpha > -\frac 12\,,$$ where $$\label{eq:`
6. `act=4.54` target=`'E'` snippet: ` {\mathbin{\oplus}}E^N$, where $E^N$ is`
7. `act=4.54` target=`' $\\'` snippet: `tau\,, \qquad t\ge 0\,,$$ where $\Gamma$ is the`
8. `act=4.54` target=`' $\\'` snippet: `where $x$ denotes a $1600$-bit input and $\oplus$ denotes an`
9. `act=4.50` target=`'$'` snippet: ` probability of the $i^{th}$ state and where $ \sum_i`
10. `act=4.47` target=`' where'` snippet: ` $\mathcal{O}(m^2 \cdot n)$, where $m$ is`

### Feature #2299

Top 10 max-activating snippets:

1. `act=6.37` target=`' ever'` snippet: ` this example among the most sought after and valuable American muscle cars ever built.  `
2. `act=6.01` target=`' ever'` snippet: ` an example of one of the fiercest and most powerful vehicle ever constructed for the roadway`
3. `act=5.81` target=`' ever'` snippet: ` ended up as one of the most anticipated ThinkGeek products ever. The iC`
4. `act=5.67` target=`' ever'` snippet: `, and some of the rarest and most desirable muscle cars ever to come out of`
5. `act=5.50` target=`' ever'` snippet: ` Kim Davis and say that the South is as backwards as it ever has been, and`
6. `act=5.50` target=`' ever'` snippet: `ian sports newspaper.  Senna was the greatest driver ever and when someone like`
7. `act=5.49` target=`' ever'` snippet: `ivity constitutes one of the most fascinating fields in condensed matter physics ever since its discovery in`
8. `act=5.49` target=`' ever'` snippet: `  “It’s the most time-consuming thing ever.”  “`
9. `act=5.41` target=`' ever'` snippet: ` knows that 80’s action movies are the best action movies ever! Adventure, explosions`
10. `act=5.25` target=`' ever'` snippet: ` sex and I had the most powerful orgasms I’d ever had doing that.`

### Feature #12112

Top 10 max-activating snippets:

1. `act=8.34` target=`' pay'` snippet: ` in the laboratory.  It makes sense that the bees pay attention to sound.`
2. `act=7.97` target=`' pay'` snippet: ` boss DJ Whatever you brothers weigh that's what ya gonna pay  [C`
3. `act=7.95` target=`' pay'` snippet: ` competitions they entered, meaning that uhlsport may have to pay out a lot in`
4. `act=7.94` target=`' paying'` snippet: ` out of bed in the morning. Yes, curing cancer and paying the mortgage are incentives`
5. `act=7.93` target=`' pay'` snippet: ` Canada. Like most expatriates here, I teach English to pay the bills. I`
6. `act=7.86` target=`' pay'` snippet: ` parent of 5 kids, summer gets expensive. I have to pay for swim lessons,`
7. `act=7.79` target=`' pay'` snippet: `field, Amy Winehouse‘s goddaughter, attempted to pay tribute by singing the`
8. `act=7.75` target=`' paying'` snippet: ` said. "And the parents know this. The parents are paying tutors to go`
9. `act=7.75` target=`' pay'` snippet: ` your iTunes. You should see there wich Apps you already payed for.  `
10. `act=7.74` target=`' paying'` snippet: `, that acronym!) for free might be interested in paying for AMPP,`

### Feature #1689

Top 10 max-activating snippets:

1. `act=10.89` target=`' palp'` snippet: ` are prolific and the urgency with which the concept is discussed is palpable.  `
2. `act=10.60` target=`' palp'` snippet: ` the persons and weeks, but criticxij for repairing the palpita of the paral`
3. `act=10.48` target=`' palp'` snippet: `-the-evidence ground unless it is `plainly and palpably' wrong.`
4. `act=10.47` target=`' palp'` snippet: `, because, they contend, the jury verdict was plainly and palpably wrong; (`
5. `act=10.28` target=`' palp'` snippet: ` trial only if this Court concludes that the verdict was plainly and palpably wrong. Generally`
6. `act=10.25` target=`' palp'` snippet: ` thus, we must conclude that the verdict was not plainly and palpably wrong. Stokes`
7. `act=8.98` target=`' palp'` snippet: ` processes including inspection, listening and smelling, inquiry, and palpation \[[@B4`
8. `act=2.24` target=`' phantom'` snippet: ` of time servicing the false interrupts. And rarely a phantom keypress does get`
9. `act=0.84` target=`'able'` snippet: ` prolific and the urgency with which the concept is discussed is palpable.  All`
10. `act=0.65` target=`' blat'` snippet: ` it.  I have hurt while watching voracious and blatant attacks on social`

### Feature #11131

Top 10 max-activating snippets:

1. `act=5.05` target=`' in'` snippet: ` this from Rick Santorum in early June, when he said in a radio interview:`
2. `act=4.76` target=`' in'` snippet: `-doubt that had kept her from doing, she stated in Starting in the Middle`
3. `act=4.59` target=`' in'` snippet: ` the United Kingdom and the United Arab Emirates, he has said in a statement. `
4. `act=4.44` target=`' in'` snippet: `. He really wants to talk about Stain. He says in the few minutes he`
5. `act=4.29` target=`' in'` snippet: ` with Mr Johnson on a visit to London, Mr Bolton said in the eyes of the`
6. `act=4.25` target=`' in'` snippet: `aughtery, chairman of the Wayne County Transportation Committee, stated in his correspondence to Sam`
7. `act=4.25` target=`' in'` snippet: ` Brazilian Federal University of Minas Gerais, points out in our film, “`
8. `act=4.21` target=`' in'` snippet: `uhrenco of the University of Surrey and I explain in a forthcoming paper in`
9. `act=4.13` target=`' in'` snippet: ` confirmed by Status Coup.  Clark also noted in his video that G`
10. `act=4.12` target=`' in'` snippet: ` our discussion.    Mawhinney further stated in the memo that:`

### Feature #12181

Top 10 max-activating snippets:

1. `act=4.40` target=`'.'` snippet: `import android.content.DialogInterface;  import android.support.v7`
2. `act=4.27` target=`'.'` snippet: `  import android.app.Dialog; import android.os.Bundle;`
3. `act=4.24` target=`'.'` snippet: `import android.app.DatePickerDialog; import android.content.DialogInterface`
4. `act=4.20` target=`'.'` snippet: `; import android.os.Bundle; import android.support.annotation.`
5. `act=4.13` target=`'.'` snippet: ` android.view.animation.AnimationUtils; import android.widget.ListView;`
6. `act=4.10` target=`'.'` snippet: `; import android.view.View; import android.widget.DatePicker`
7. `act=4.09` target=`'.'` snippet: `-548 MSDN: http://msdn.microsoft.com/en-`
8. `act=4.04` target=`'.'` snippet: `import android.view.ViewStub; import android.view.animation.`
9. `act=3.97` target=`'.'` snippet: ` in msdn. http://msdn.microsoft.com/en-`
10. `act=3.95` target=`'.'` snippet: `package other.writeily.ui;  import android.app.Dialog;`

### Feature #14671

Top 10 max-activating snippets:

1. `act=1.60` target=`' or'` snippet: ` chicken.  Beaver’s tails—believe it or not, this was`
2. `act=1.54` target=`'var'` snippet: `precision) {  var i = n | 0;  var vf = get`
3. `act=1.51` target=`' or'` snippet: ` the gym. (via Heat)  • Come hell or high water, Cow`
4. `act=1.47` target=`'for'` snippet: ` she's at home. There isn't a clear like-for-like right w`
5. `act=1.45` target=`'说'` snippet: ` been in touch with other companies about updating Continue Reading﻿文件说明：  `
6. `act=1.44` target=`'z'` snippet: `Gambar dierong shot mcm best...coz we love it for`
7. `act=1.42` target=`'*'` snippet: ` us as a subscriber.ms in 72*z + 56*z + 10*`
8. `act=1.42` target=`'Or'` snippet: `clerezza.commons.rdf.BlankNodeOrIRI; `
9. `act=1.41` target=`'�'` snippet: `μερώματα της Δευτέρας ήταν στην Α`
10. `act=1.39` target=`' or'` snippet: ` graduates.In...  College or work? Gap year or victory lap? And`

### Feature #8970

Top 10 max-activating snippets:

1. `act=4.65` target=`'Type'` snippet: `sql, conn);             da.SelectCommand.CommandType = CommandType.`
2. `act=4.60` target=`'METHOD'` snippet: `    after :each do     ENV['REQUEST_METHOD'] = @old`
3. `act=4.48` target=`'Method'` snippet: `", InsertionMode = InsertionMode.Replace, HttpMethod = "GET"`
4. `act=4.44` target=`'METHOD'` snippet: ` . Tried :  RewriteCond %{REQUEST_METHOD} !^POST`
5. `act=4.44` target=`'method'` snippet: `         this.saveUrl,         {             method:'post', `
6. `act=4.42` target=`'METHOD'` snippet: `   before :each do     ENV['REQUEST_METHOD'], @old_`
7. `act=4.25` target=`'ARCH'` snippet: ` # # Darwin #  ifeq ($(ARCH), Darwin) `
8. `act=4.17` target=`' mode'` snippet: ` <- renderJsonedit({       jsonedit(x, mode = 'view')`
9. `act=4.12` target=`'PORT'` snippet: `1 [R=301] RewriteCond %{SERVER_PORT} !^443`
10. `act=4.12` target=`'PORT'` snippet: `1 [R=301] RewriteCond %{SERVER_PORT} !^443`

### Feature #1433

Top 10 max-activating snippets:

1. `act=4.62` target=`'\\n'` snippet: ` sorts of scams that don’t even exist yet.  For example,`
2. `act=4.49` target=`'\\n'` snippet: ` people never realized it, until they were told about it.  In the end`
3. `act=4.28` target=`'\\n'` snippet: ` investigating a maximum of up to three authors at a time.  Briefly`
4. `act=4.25` target=`'\\n'` snippet: ` to seize what his advisors thought might be a bright moment.  The contrast in`
5. `act=4.24` target=`'\\n'` snippet: ` Mr. Trump that matters more than anything or anyone else.  ANNE-`
6. `act=4.13` target=`'\\n'` snippet: ` more like reality than all the talk of impending financial singularity.  Markets seem`
7. `act=4.09` target=`'\\n'` snippet: ` needs to change the public and governments’ perception of it.  This advert by`
8. `act=4.06` target=`'\\n'` snippet: ` it like every other part of the country: indistinct.  Faced with`
9. `act=4.05` target=`'\\n'` snippet: ` what drives market outcomes and reduced it to a seamless system.  Many believe that`
10. `act=4.05` target=`'\\n'` snippet: ` learn from experience.” Bees are able to do this.  There are six`

### Feature #9712

Top 10 max-activating snippets:

1. `act=7.11` target=`' resource'` snippet: ` a cool side-effect: since the check isn't so resource intensive, you`
2. `act=7.10` target=`' resource'` snippet: `Q:  What is the Java equivalent of JavaScript's resource folder?  `
3. `act=6.89` target=`' resource'` snippet: ` each having 5-6 icons on them. And, the resource manager being a little`
4. `act=6.72` target=`' resource'` snippet: ` specific instance tracking, just a simple decision based on current resource usage, and a`
5. `act=6.69` target=`' resource'` snippet: `, they might differ and in order to keep up with the resource needs and to ensure`
6. `act=6.67` target=`' resource'` snippet: ` is to make sure to load images only once when using the resource manager. Not only`
7. `act=6.62` target=`' resource'` snippet: `es a VpnSite. // Parameters: // resourceGroupName - the resource`
8. `act=6.60` target=`' resource'` snippet: `_refcount(); }  static const char* resource_to_c`
9. `act=6.52` target=`' resource'` snippet: `itesClient) CreateOrUpdate(ctx context.Context, resourceGroupName string, v`
10. `act=6.52` target=`' resource'` snippet: ` err := client.CreateOrUpdatePreparer(ctx, resourceGroupName, vpn`

### Feature #6941

Top 10 max-activating snippets:

1. `act=5.26` target=`' {};'` snippet: `flv'};"     + "var params = {};"     +`
2. `act=4.72` target=`' {};'` snippet: `:  'use strict';    const obj = {};  Object.`
3. `act=4.68` target=`' {};'` snippet: ` = 'true';"     + "var attributes = {};"     +`
4. `act=4.67` target=`' {};'` snippet: `ido como true. Veja:  const obj = {};  obj.`
5. `act=4.63` target=`' {};'` snippet: `:  'use strict';    const obj = {};  obj.`
6. `act=4.59` target=`' {};'` snippet: `    t.plan(2);      var files = {};     files[`
7. `act=4.55` target=`' {};'` snippet: `    t.plan(8);      var files = {};     files[`
8. `act=4.53` target=`' {};'` snippet: `    t.plan(2);      var files = {};     files[`
9. `act=4.52` target=`' {};'` snippet: `    t.plan(8);      var files = {};     files[`
10. `act=4.30` target=`' {};'` snippet: ` (cb) {     var fields = {}, files = {};     this `

### Feature #522

Top 10 max-activating snippets:

1. `act=6.91` target=`' academic'` snippet: `’s hard for me not to support something that would increase academic performance.”  `
2. `act=6.83` target=`' academic'` snippet: `And the concern is how can we have the graduation rates or academic performance of all our`
3. `act=6.60` target=`' academic'` snippet: ` I participants, yet they account for more than 80 percent of academic infraction cases.`
4. `act=6.34` target=`' academic'` snippet: ` Mark Turgeon said he’s concerned about his players’ academic performance as time-`
5. `act=6.14` target=`' academic'` snippet: `If they do well because they spend more time, get more academic advising … their freshman`
6. `act=6.03` target=`' academic'` snippet: ` the best way to accomplish the goals of raising graduation rates and academic performance for all student`
7. `act=6.01` target=`' academic'` snippet: `-19A cell line.Food insecurity affects school children's academic performance, weight gain`
8. `act=5.94` target=`' academic'` snippet: ` food insecurity in kindergarten and 3rd grade. Children's academic performance, height,`
9. `act=5.86` target=`' academic'` snippet: `as of Appointment  For the 2015–2016 academic year (beginning`
10. `act=5.86` target=`' academic'` snippet: `s what I love to do. I love to solve complex academic problems in a complex`

### Feature #490

Top 10 max-activating snippets:

1. `act=6.67` target=`' missing'` snippet: ` printed ear for a number of young Australian children born with a missing ear. He studied`
2. `act=6.52` target=`' missing'` snippet: ` name in the top right corner.  Fill in any missing fields and make sure`
3. `act=6.43` target=`' missing'` snippet: `University of Pennsylvania Law School alumni Category:Year of birth missing (living people)`
4. `act=6.43` target=`' missing'` snippet: `ard there, an empty one. So it was indeed a missing sword — a fact`
5. `act=6.38` target=`' missing'` snippet: ` few weeks ago, Susy started a massive search for her missing friend Geno .`
6. `act=6.33` target=`' missing'` snippet: `:American scientists of Chinese descent Category:Year of birth missing (living people)`
7. `act=6.32` target=`' missing'` snippet: ` people Category:Living people Category:Year of birth missing (living people)`
8. `act=6.23` target=`' missing'` snippet: `B Grade items may have been used, have damaged packaging, missing accessories or a combination`
9. `act=6.14` target=`' missing'` snippet: ` births Category:Living people Category:Place of birth missing (living people)`
10. `act=6.07` target=`' missing'` snippet: `  Category:Living people Category:Year of birth missing (living people)`

### Feature #1545

Top 10 max-activating snippets:

1. `act=3.61` target=`' push'` snippet: ` your registry as a prefix. For example: docker push localhost:5000/`
2. `act=3.42` target=`' push'` snippet: `'`). 4. Push to the branch (`git push origin new-feature`
3. `act=3.33` target=`' run'` snippet: `  I can create a container running a registry: docker run -d -p`
4. `act=3.31` target=`' ps'` snippet: ` my comment below - this registry runs locally and with docker ps | grep registry:`
5. `act=3.02` target=`' pull'` snippet: ` docker push localhost:5000/yourimage  docker pull localhost:5000/`
6. `act=2.95` target=`' 5000'` snippet: ` a container running a registry: docker run -d -p 5000:5000 --rest`
7. `act=2.90` target=`'Pull'` snippet: ` Docker executor with image node:11.2 ... Pulling docker image node`
8. `act=2.86` target=`' --'` snippet: ` a registry: docker run -d -p 5000:5000 --restart=always`
9. `act=2.84` target=`' -'` snippet: ` can create a container running a registry: docker run -d -p 5000:5000`
10. `act=2.80` target=`' 192'` snippet: `.  192.168.0.3:22 192.168.0`
