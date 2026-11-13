"""COMPRESSION GAIN READOUT (inf_compress_gain, 12 Nov; readout only, training untouched) - can the TRUNK tell the store which patterns are
real? The minimum-description-length test (Rissanen; de Marcken 1996; Schmidhuber's compression progress; Spens & Burgess 2024): a pattern is
real when the trunk needs fewer bits for the text once it is in place. On the embedded Re-DocRED documents of the domains visited so far,
scored against their gold mention annotations (_BENCH_MENT: char spans and entity per document, built from the Re-DocRED source files), store
and memory off, no carried context, each document read through its domain's slot. 'Thing' stands in for a removed name.
  (a) UNITS - BINDING GAIN at each join inside a run of capitalised words: bits saved on the second word when the real first word precedes it
      instead of 'Thing'. AUC same mention vs mention boundary, against the plain surprise of the second word (offline stand-in trunk:
      surprise 0.565; capitalisation statistic and string cues already strong).
  (b) LINKS - MUTUAL COMPRESSION GAIN of an earlier mention for a later one (up to 8 nearest earlier mentions): bits saved on the later
      mention's bytes when a snippet around the earlier mention precedes its sentence, against the snippet with that mention replaced by
      'Thing'. AUC same entity vs not on all pairs, HARD (no shared word) and VERY HARD (+ letter-triple overlap < 0.25) pairs; the
      letter-triple overlap itself as the string baseline; antecedent right (best-scoring earlier mention is the same entity) by gain, by
      string and by their rank average.
Offline (tools/compress_gain.py, tools/trunk_guides.py): with a small stand-in trunk, trained states never beat an untrained network."""
import re
import json
import zlib
import base64
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import _fail

_BENCH_MENT = 'c-n-$Nw#Cn%>%d>k2Q$Csjc|@E2+&_5Cj+@z4WP(U;4{Nx;GjUMDzdtC;jLD`R{-KJN9SxPx`;le@m&KQr0J*pKHsX+Rv};=Y4+O6Thd8@o8gy>iM~s{pqcIn*Dr!Pd~q><LCVuseZ=!d8G4m9-nDH=k*z-eAfASrumuY-`n<c>E|P~Pqu$xrT$qZf3kgwRqPl0lkH=y+CHnEk7xh(8s``5{2BYl_xa2A`OCBY>-+o*yNpjB|95`%-*o<3eDYk*|Ap^%{=(Yv&CXvkGrr5LpYyjh{LSZ&<9p;<&)+}rJ@Pmo>3nbd9{K#{ycWJkN##?{zhK{^oPVO6e;v>A{Jmur|4wDiPua`AQ$7Dq-NV0A>*w*EPl4~$&)4YoG4Vjpf9bRMck1WQ*b}zjHO|K$=ksSg-hCcj8vZxWnm_Zrk8B@uo)13z<U>7w*&5Y;@%>E3rzC$S=i_cXlb+8@@_*?(orz~M_h&bMCh7csdyn>WJ0HC3nVg@`qqAp{&nGR(-)%WBt+eGIqn^)Hd*X9A3$L!^&!L|0(em`?(9YUx=VSPK+n;vUl|Peo7MXopJd^Vs`dLAE4*h(LewH!5dOsi6)?!{*fam+<^Wp5%jCnr!>i8xzZ>+##|I9r9oGrnE=V)Ji_&%8H{D<@13s&HHO127%ufY9Twi2-B+Gn4qU$7d_@7U)x*q_+vGwrif&Y$oHf3_E%MEacHg~xE7aMpuA?+Xjzyy4{!{;cxM2Y<fMSzNvt&xbop0uTOtQCoyqXJ>bmR{iIG-tF|wR?cfIYx@ham&1)(&-<^%H*-A;yz*ad=kIHkYp0#n(9Sc%OY7&M;i2OLSKI%?_y76K-BuVr|9LEZ@}-U3Kch{5X~R~Rtx7E3^G}Y!mp0GqoVIOoI@qc@|1MtIJjcm84;U}XPJ?wm8eY^oA7C|qo%<X#`y3Z|QTsgKZ4U!4r=4FkKjQZJ=l0p&c!S$^VLA&MuPvu)zntf>`SZ55na)!#evsH2@F#m7ns1tOKG;XWqxcMM&+}pNNb>o2^I7fq+s<Oi%eR%iq0;;fl=H~SOn9s1Jmh*#zlk?<em>9I9!9nEs`}e&=U-{(JK!yyJ<}7nRzI&2FDC2x+u?&V?mW72UNRp4`QW4SoXcl1pMS=WQ`@@d@AKpIyvsd!oaR*Q6ti!PrI*jA%Gu6jtUfHoY&#D-za%@XbM-fOe(JVpn}3pAyPe*)p12#%8pVgVdjUV4N3s3c;-EQuz4*Jp+OPOaZ1wn8$mcT3yZX<z59}e}B0F!mZGWo$tcmG|RI#J?oCW?=6W0629z1`aeJ`x{^Q>Ci{v_=fYLid0oxi3n-(LM3R^1*-{qu{ny!+rOKF<3bxoGj9RWg!J?sKr%h2)F>d_XQQJ9+F9!~<RSX>C_*e<Zd-%kZsmezKR}xXg31lQo^sSoq9rHJsCrXV3Y5Jbe7oZa;|5yQ;-cv3j;ot>vRzs@c}SabSnDKmIoDJkH$nXPFJ&`DfqsbIswaorOQTFXWl*Yx-@o&RMaB-<!)$wzI>kuJ7Ute4iz;XWUfh(&gto3%Kw^{@8(vQ}--WUlhp-2HzItEQqrGF01GNw=Ly|-TB$}0$zB*x5HLD3(kICc=zX9pQ8d>&9?FOW4tFj_A<BH`D%UW7w2#4i~DYzbzs}e_nmF)^R%!ihMm@li(;PIVA^-|U!4E4cswrK@`ZQ%I@jy6cRF=x`x$1pqu(88+b!QA=Z&1Nj7M=ke(wJNmUC(p9?*6Pm*J;TSrvBx9>w`owg}?XDSq+#^*s1k&P#8Jk0RsL+3yvcq8jwe%5H_OiyoCsU)<^EPgvQ8|8NdGs|WpYox^lgf7#<a)NvLRA6>?s?N95Ro3nF2&9naI82+^UqGs{qXPZ~7JDuw|u^QOfio@*tX<J#d6OR~sJGQ~Nw{WnQ$_;}RX^X*k8fv-o_u@Zht4VXS7037FPcYko<i%C78$<VmFIoH*yeRB|S-xWJI4j9tZ?R8vUJYK^`8{Q_eziTmw0|4+?}+`&3&j2zTOU~Mr>xjsqk5s0c+{PnsCEB5R#(~k^fz-3SzG3Q&Er5@y4p`2c>ZmkZl^f50ura&S(J9Y<DH#XH2TmrZ*Q||9mkeFzrXtzwOZ|b4}AZ#H}d4Jwq2b3crWK~E?t*q#<Re8J|Ee3HNJB>-?J@#=j7ET&W`i^PYvhaf8#9UQGfPEW3SQ9snRxk4PWJ5b@}qTJSuUW?&mL_=L7m`JiiMEI#%O31!f&jDQ-0|N}FMuAEk}27QX`ZzjirK|Cg-H_b#Pdmx2@2?kL`a_O8yV!UD2?sr%L|=V!b7a@>7f9Oq}N@$iqFg;6*6{CW88eA}hxamv`O&V;)b701*=JAbejHZxzpK&hQ{{OK&LQTF*?HtlUSzw3OT`f^Y)m)p-avTBLz{!@LISoOrd?UY;o5bUKS)yj4t`z4jF7|Fc&Y(bs>A9aECKGheF*UhYzPi;|_KZ9bY>sIq(4_>WSJull%P5g)R#H?NRM&>+vyIt_LvDxe(LAhqT-Ol!or+b#+;$mrb1D*AW#bQga+vi_6QwPpe99P}GwQYmAg!=Xash`hmI~4zAoI__=iRlgQ!h=V>JI=pj4I|3kd5-64MZ4CdHCj<|G%td!^jN4D*}bp=K4*@%M-4rT7HRcYVz<0++T76)V_jNz)xB{n=Y!P5^>+%6TK%4>!)h|W<s5YIvbCMRqqX&{%fYpH?ge{#erVYRRDBx(emIBz()ly|o38Vi*Jg3I&&u1C_s>3GWjC(ZeV(pW!CnrZryJ$)whEPa$+l&({3C8GzvOG^Tn+ZC&UQFk!(*U9VI#qt={&>cmDl;+!>?>ZW4sB>cCc375Z(G*FK)A)YTHl0Vr_fz6W<rE*D<zeYx0V!p2c4)FKN3TYEuV3?TY7#*X%6Oer>j=78ma5R?aomu)}LuEU0rVSxfCl-noSgfr$O<d(MFc{Bv2|(%Hplqx+Y%p1*T<p4F0kGXqvh{?2?Av6{CgyH55|=Wnx83P-;ccmBSuYN?Br{rP%!y7_8J)@xTk>g+Q#ZzJ0Ii?A8*HBfZCb!<lLxKSNq!2PfcV&I)P&fjUf&Yu=`oc9=G*@oM!?YNf$k1@mqb)IdQ)mW@p1LS*a8?0o1?ZoL;Q6qa5V`p?_SKtfRZp!cLb=cj6zv%qk%OAXTc!_nQwJ7ZO4Y(;a5kv6a$t%sXy~^Mow6@6c2RtwWKd1ruQ2SYmwgc~J<$J&PhL#9OjWrB>Yvb&I(R`WO4jqd<7+|f;s@j1X0%r4kJYa=2kO3Cb%U*xwWxUQq!DWI6e0I;`W<>vS);+4kvwxR05jx$tEziZ_Z@_LXe{fbwW!}5?u_o_b+Z6;0-(tlYlwEk%ox?KuRc7_=Xby(1bCoRKAj?3}q7Qg>&M4jtoik%w>FcM>>5?{!m2KsWtv}VAtlCQ)Dfak@O>FxXmEXb-6sraoKeyc*lgCL10PKD9LXkD2;N<;yK2KpWRjmC=N%vz#bL~e9kR^e8=M|x^#GQDaS*<);9A(b_&J1p@!iMTO&8i(O*xlABTGy~*7Z;Y0=lc$#4X9Ga=KW~ptXgAlyxAC!v+r!7@k-3km539Jzipmpw$5Jhdr(+OEHuOSqDkX+S*NX9_Bdzk<g>4^VXa}pNvN%C=Wr4@W!IiJ7FItd`)}n3sGU&+rHYfW2ym`Mm4<<fT7Tz#=ft+x`a2ev-HH>|gMgtY-Y@FD;qTYb=$`!jcDuV<w7u1Pf~+<m4#V@EN8<Yx+~kL${bP$Y&p5wrChkeA0ImG+0(r1^TMx*JcWQV)o^|@FmiLF>$co<eT%p)yh8_B@J;wv@8LU#Cb~$4Q`~71zz6wo)FTKUEU7d$Nscb7g`H6)Kw{j;mg9-_A+x0!CF4=}lHvN{}=P#_pO8_|~p-fh*YUW|Us~xb!pX|*{;ZJrl6<yKiFE9Ypp}^^7I8NJYZ;dchK(QAX?*hh`LsK|?&p&9p1kdk$<S~f1;tE^bRKTok{oz<Xi)5C>-D3E;ot0hyS`jn3snC}v-$(Y6S8Cik=L1-^iS6RJZf7T88*JQ{PRzU-QtprFEKxt(iu!4s#k|n%ZAIgJZUO6qaHdTwk^J7W^2ko86=xF5B|DP1p{<F=OG<i4xvO(LXHEMm8Ttc{sEf4EkhgHb4j3c{T)w$6Y*t~t(?!S!#~NOw&kF&S7FQIz#Ad*A@lgRMM7ZTyI9fZ#lfdAH%;La1Mvb;?t@ud*oO*iof^lHwNskVRU*0hX|5vMF8)a*xQ~c1zYCZpKE36b-Z{9&RR+l|#w1ukOn|~W70y_0V!4Sp7m1|}8<_&-&#$aGyxz_G1)eYyuN_-v$Y8mX3{%1dV3ge-IwEHn?&%__yv`5dwV4zg;!@xeo<Y8c4mKU8~n06ip=jXwS+wM^7cN}Z8zK6A?p%!)#j(0GI4`Z$EAogJAMAaz8>T@exy@fOoe6aX-XgYsG(=+f%j#Yd}yZPWye5q-)B7qqEIA7a<OTRjSzWJ&fr`aC{v*Ynv|8`FEJK6fJKy36upmw4i_wvaKenR&48@s<PACY<~g^^V4d@?pEBfewj=j4_(2!lnw9i6ReXX5x|8Dd-1Mkc->Z8y(2KCR8xT+yal)!}1%SXgP;jOXpzyw!0m+iRUA$-sTX=YnU%XFSfIwVTeL@p;&`4t>T5kU#VEJo{#<OZQqZajs~!h85xXFXzy(aur9ZH3=vwy2dB|#Lpk}o6lg{rDxAe=d2w<i-r*|-g@>(cpKU!JX8xh_;@=l;v<G5jrq(HUwnl1U$GNVt+j(Cgn|r#uPkJ6)DG2xR^$!!8XLWapBCrnh(&MHhBzwEm+hKG;SimyA^Z^K3I@?Yr^T@4oHYRE+V)}7IW01Vz8vuRW3<VUUo-<{e{tk(SC$gMg8kuHThiD7*wWA`VMy)T54gk=$F8lx0aUVK)bvZoPSF^++LLHqD|k9V04vyr@X?NTPy&d&H2|?aF6qn;2~*VvRGT^jGs`C1#lAf3JthLLhB7T;I3q9|qb=B|WW-R&p$D7XH(pKl$?g5&yfKKkJMir7b7E^+y+-uiQ&C|=7p5XWlRzASi#NX=&SUZJFW9v>wrxJtIF_<wWIN{I2!)qiJ}&T1XP$L{$^;e4c~RRDQ0vGRDG{~_trS5W+N_l1k-lsz1e{~1HYp!*@?oBdW@D&tJVoSY?=^w^`;|O!8e+{@R~s9#qbbAsv0|Tvioqa-g$f$9LDr1dpR7l$RkJdO&6gn~oTUI7*=!IEj||F%0Zolv{QLZDFAO$5-lCI!F|3YCysHiOo+auO8(6azA4aP|Z)5fS=3FY-Xry`Hg0IBUoz0jy&o>OBumUgk#wQ0GhP+nQZ4<@yRx8+jHF#U73*o#iJNEnU&sKpi->+XkZ>vs{0HEg=R<rb+H6eI^VTE4NxZvv-=c%9=$I3k46Ti4Kvdp9G>ld~PX43`!R6bVEOjSPXY~kevj-5~EE=>OQ!GNkHv+Qo@rFaomB<}1N&8ZEov^8ATOGP`tlFb~*H?jX}y^qiRZ4KGnzxdqWnkAcj6+)bL@Y^EvxxXFJHi@XtyaGDj2%6X^CmTJ*Iac-zYu2xTz!E0T_HHvKIoT)w2DmL>2ib1_Y>PR1<s#-wAdV&agup;D=6pQ9tu`;Lhz+uB+<x{CZNCSxW?40BWA)#;sU6>(ANbsqg^8Wyw=mh*VwYDC$GR0JOTSnoJbBNMyv=Sxj{=9an`Q@>fHj9to2rjUv0%<`nVU$V9c<vv_%OcUL~|m|dFzITdCg%*RpKH6lco_M+2<8*D{e0mzI=IXJHYKp0s6qm#A-ta!!t&y;*a6nxB5m0K8G__xA|i>M9p7dwo9C#F;MiuJJ?!c^7RZ3HcCwmtlNh`KK1h!$DU!|KJNz$p>MJfjP^q46|^mHcMMN8REEHtZJJ4B-%iZLqCXpSkg;;IpXT;UpCPiMNKe1U+CqTE_-OWQ`;bSMmt!2Cq;=a^fZ!Zmp)pIWE)0mu99>OZ0$aQeQCftK`!B51=^APcXTkU{Y^TkpU)Va<pE++7FuGqH3%9+Wd^7u5RW%RJF&sU=*rLAJv{A$c>u6{2zP8m^Sgv2g5i1>do1T)*ACy7b{T5;Ggpb7!UPx0BRzOLdS|ptaw=MM5FAJ!M1(bk}@#~+Jr1A<gQWmg&Q>vw~{-S5~xD_axUlrE*549IYUJ-54s91r{;MivcYZZ}ovtG5H6^#kHw-E(%1Xis^B_Xm2%IBDBBPyt~q$gVEux{sINVl)^;@mhNTP5JgCCIX5JQ$E$CO4sh7L`l2<N220a*zsa_vB^=g-V=+vz$Qx9Z{7pBLC+bRUg%ZF2w1c69-fo+8H;y%0z3jG-<ZJDiMC1X`_tyg4d?YmB;6_#<6oSGmVin>PxgKnh%`)6I59gcU+vyF!z+TIFt6Xo%c&nloscm+~LD;aau#ByUxz*&636_YQhnA0gUboasY}-nk5h4&uIO5#Q+DK+$MkllSqKsSDo_*#?ug0yaqCD3_cON27N4xwOzb(z~i#bHd(89xy(M;YUO#SS{0M5m{DnI8v}MXKich@*cx!3Bh+r-QtneFu?h4Qzqk!u^GW~Eq~DMk(jAIJeO=FYh!$)AgnwhhXfiuCRM6z0i+s6sTab6=I4!3Nm`@7!1LKlCs==(mk+JrJ<XH^52%&WwHBXJ9x08=ctp6&Q-02ZB2GoMgF_8e2`^6ZxmOGOf2~wr+N86*V@lrIt2`d}xmq>K>N5Mw~wlE(BXf&)^l9S`M_@*t+K_902$$$Q2(^SXTm-uZsuJ(KK`!k-_BGz=j&Wx$>5dWDqn)M}#Uj?kN!5w553ct$qd*6|`g|FoUssvENchd&B;O?eOVAqf&j`+?I^L^9B#1~zW*!j5Wxao{56&!@>u#$77Y|<13n=J&&5_~0yENvx-cz$+Vj-=`hXP(cuu^s2okvCA|wdm>{qntVWfrtC^Jjz+g7qJaHSG(k*hv#2^@Zz&trmx;pC~YnqkOxVk0p}odXAQu@-j8-PZ!5q7{{Kc}@BJBvXbG8Igp)*XK-qxRhiD7jJQFss?znh;<4t}M;k-d`G6P#Edh9-6o+86cy#mlu{R4g>d%k)bH7ZuNh?IVAhEaUEVUso@sl$ffY!HIUyPoaa$@4^6ey?r+4}M7r3ACePYK~$%B?S`LF%`z;W6krNA_<%eiyPXA+S{R5J9N(#RXHKrh7B}7hxWPAtVVyy?QMw9n8|JL-jUhzwUXc4E-S9YZpy%=e4IsoK?{I}2NK&T=Px!uM>X^?n^k&60oJ3P_X7znJYOjm3K=YR=7_Ur2q5MKxZ|w6Z;nl2L@*mM7aKzIz1e81JTvz$oXm*ei3gYwtqlWm0=!KfFO|SM>o~N?jj*Ccf2&oZnfPiIfVSu52o#MQIv~Ot<i^eu+VBuET=4oy=`m>%f4?f#;O6S$>c%T0P1zo>iWfUzdhnqPE!>2GtP-%t;ujHVn=!AgVkT1)<IksC^~9m#wD*#<7|)Ac;;tL#!r(F9=2@fJmfixM1KDD5P_*IVzk&|H!E5vW2am!gg_$QjCXH&b9X7tBQy$A!HVCBi!iZzkZf#fB_sT{<RJ~{`V)^GpEhB1z0enkf7j_)aQfK0jx`(E5L}lZIe3tO%C){XL?0tN+QN=I@KsC1D=Ron#gCksQTvnDdwk6BDHF)3LVhL>_wpf{%P4|#pvDCx_l#kKS9s<zC{_QCM+9#mA5ZiO-qwr*zgUOOQKipv|BeA5afQsUcOb;EJXt5x_s`FcRSuF`2GJw}1)nDag8bZG*^^Y+u7DrRtUVG(@>gDDZ%w=0~h^>v;f7*e1*>i80seH2<Z2&=(`9tZWEnTu2?Zc8GX%m&pGfjkV4S%T=!*!G?i`*)Mb9Zp?hgbqW*`g@0f{N$mXLa>ej-geslA9}js%}YbT(E76M#zBI`4?c!*Ndpw8y!`W0yFbq1ocIK1RSeMwk~>g5t(<wRrObzVxxQGToGETmlI=DxX!iYge%^APQ@26X*`PVe?&bg-&S*t3lSD8{Y_Zq?=nM|wzmGO?RS97Y!mFHcf886g&hJ4wYK@*Toqs=$pjlGQhU&u9xBky`qLN+<X&1iUOKs#8=8|#4W87KOeglHAR*zP{%(Pav}J}&aqnZu)Cq!F(MSx0HGYetj8G9-TwPPVMtxZ?U~UCpvQl5xaumh@G|0#U({^*Ux;B-gC!iKfB97r#PTVqCH;W1Xs#Jhz5UMy1-MfXM5zXQes@8#{RI8bId<HoStFqZ}Ud<_Vs+0syyDmB5z%VD+<-|TwWt~61I4QN`WT$fH*dphenuX2thr$ahc_>qW+Kca@Is3keOR<PWp%7=8gXhgM2c$P~RnH-?`Ml4DYC`E8S<*8;f4Z{|)TE0vRFjvfr9AI?aFP-NjN3ctX`Szi=D|6;=LxI2wu;oqIex1fR;&@K*p%&$qMC&f^jC?PtvwpSMR)|9{A0MBp5;oxg6-zcC2B?S;$uqlN+7$d(RWP#MLINomxj(hk`(>Kq+Hd{V=oMXiqgG#Yr-Wt_4IKtZ+twNF2EA?I!RPulS&SQvRuqHTSnV@v%hS4?yj!EvZfG4e6+%cv|L)=;!t6Qc)qYHcb!o>F<YN;Z*wgf2;QAJYE#;);00Y*=xz6bvX8(jSR=lcjObfg4jV#g9Tv$-#wBGTX!x%6NFgT`;N!$+3b4SH7~IELawKz`4hGYS9lC?2rf7f!=vN51VZ7*V9WL=QeN-V0_BIebJ=2L!#^yf1ZClleD$PLgHKQJ|E5-11!ckWCSiQ8KMoDFty>$az?zCbHz^I+-A{7hPEbH3Z5S_FK%jkkqEB|h&=h<IF(&jC)PXH?21&mk2;=wB5CbETbBHLgcm%XZRsJi%uaN*`Tm6Arh(4TPK<9HDoVIm%s%7CO(0{_#8j}(RZG1$Dsa0=Err)w1jtAN&=gh40Gn9gGej<bQ6v+M4+8rrtqH|pH&@lr8YNX;733K)t?RlMQcF1xU@flRjn5t(<eh6`}62QHoTKQ5@7k2WR6+?o6q3e7h4!eti?Q7W(`?^>%%egaI2=WiJBB6sQe+542hmH3M)Vm+CUzB&0P$SO?nQbOu&vjtI??4nl@Y<+PjtIt(?vPXbwj|BYP9qpJQ2={r3>pXP#qEe)u*Z$-dQ`9Pm%u3JE_Qb2#s^7U^AcHlx$elimfENWTeLx(QeU-@_07;xx@t+KHl2{t{ki0mDn$S5pm+@$Omh0e)Mv=hWb->-u3_^sYPAvK%Sh<#i6Uu22O;4yCDXyBt33?VnV=rc^SRAcho}Vf)$KZeaiwPKNS3ytPr)1;3n^9j|m)5(mJg232eQolEIsT9tZD_j0Ihl?LLokpZu99fa4J~Ak@`ZJ*tz$Cm?7l7(F%{#Sm%_8h#%C-AT^T^U)TLbFni%U}!xt+r?K*Ozzs(t^KJ&n~@Uw@sFK01gD~Qr)OzH1~sPY$v2Fp6Yj6Vp^2l{~pY>RiPcobra2fGNk-HV38Uer0gJdF!Aad1aK@#uX7?{h;+sbGcvx;&G~DwZt`a-eA9l+w=_5ghF?Z318uz{Vvw@}6E9^pq{drNA(E(D1v$)qU#v-d}j9*FTr2y;bown48c7q5<$&vdIwdz^6A;b`U_yMHF)`J>@?cS`ZL)H2&^#JK^<UT@8zK@S&pd>O#m0<iD52y)K@?b4nu|{2n`R7Ki_+r5hP(3FIb6?i=>#ifPVx)aza9^@-}IhD1qYH1KbV1pZWd<+%1X)sJ>us*pAmO-4nIf0TCyDP+VLRGVFp72dWMDPMOM8fx#ez!wd9{J3nYHc#6i)Y{Zwdln&0E+bJ510ur^Wx-xP#E4$P7YviO)DYG{2Xa!t<z898G$;f13O;(;&`#*T;tI4?>#cc(P+^N)WCdwPE#gOxm6qk9vNjdS-P>R1fE1T=&^W~h(Q33d3gs>e8Kr5)v^1DyKHF^RoXR}fQ78-S0?<rgalfJr2;QcO<AM=>lI@N7*^;xR0RIC@J)4eJ5`N-%>$B{5O`4uk-jcqVgn6sDG^GQJ%WBtNqV2`QWy<Xt-HhPz8s;<}UD}lXVakDM;eiC8I@w-<<H=IPI#AIqn{CTQ5(H`!fvbiJs<erlu1Xjr*%asv(&ypek^mA%b<meg&RQvSbi0S9ZlS0_;L{p`nfSW0rltIr;yrzP#H4a{AuP3=I8zA@CCvPAik#iuxN%$(yQzz^R*7H;=6V{y+ks6=gsb6o9vezn2|(N7JO$%T1Gw|n`Ctd~d`9a%G>qqCj2H$Lz@UCuK+-%P0!=)YQ8}NndVy9t=#Tp&nPJ&vAO=O!{+OP!i9`@LD0;hdO<7F?(g`FF&@&zAn9lW1vBp#-L(s=1qa8@^uZD5EcPXT0df8;tBNepvU`NjCJYC**t#kxl+Sxnjz4oU_2y>gsj`$x22X&STF4gNCp`D)xTha}2YVHqSpW@Kk!FH{Wd(k@*OQa$c03S*2EbM4pnM9Vg@g|AcIab8!6Vx?){5D16kX!Bjt&;`^p#`o0J2*L-D+3>v?6$b<HmkN*F$e0n#7jP9rj3sQzd6M51ye?OMRo6`)1!jtqCJlT&tz&?^8HJ+!?}xoWVc6Nx02w{v7wUCAe-9U{E?B+dW25{VyFIy{CaTeNHIoBFt1J2;E>vQv_&>i?uq30MsbynSFsh<23Z3W(4C)+)z#!-zQyMOfaFT&%X?Q0ktFC2L@AH2*(}~o_j+MQe<iBCf$zfVjC((MWV(3R;0qc?4Y%?l5aLkB)P3&WU2L08S)ZZGsoxN_DJRL&0XIi?^pN&!KiY^Y_M-hgm7Q<{i6jLfnj-O;pyqHfM+HBkhJ}{Kt`&e&4}T>J?ad8_gHkJdCq)vH7&-vaQQSv6rIX34DmhMqDaVG+W)(Z}joHNAderhwTYcEZW}ZS;j%^zrxUjw>*Bw1D2jboN?OpfJqAu5Z(A*wBrt<UyBzNM()U<FvvP2mAX#RWt7AtE>ohT0IRpA7eB5{79b_^C?Nr!A>Q9DCgCx1RWAe-)GKzdz9iW4k5A_Ch_*(n6REj3Y6FKumY5dJ$IjxhN4?d^tzeq%ZA;-$Z{1uBIp8U6VPvk+e3G_jGe*&r`BC#X&y3B!~+RMflpl@?xhII@`WE^5h#%4W!{;@lu)%UjJhRgOc?mO>JTgthq8gcufqAlhu##@R15?95OR0mI;Q%S!ds?h~9s&5A=K_U(K_y0amR#Qs3dP91AOl_iS~Z<!Vwd7<z33i`ZB3d7wgj<<YuG6HY*iQb<0V}otWgFjZ`j~j&*m5kZlMtX>xMTwP>A<luCU?}10MJun9gj_IqkuKRheKPS~a}=}NtpwN6cBQ)zEp*GJg)EGgdz7!ZMj8BhYLsnqZOb`@teIcHFisO-P`q4yxU>Wu*l69WiA-6>H+p;zjT9PG(=D>i!^>+Sm*hOj9(Yh4C;TL5TxpvEp$wF32jxZ<wD?B6=iQu^i`+$g1Z^<96ocD!zN|)_709c|^{RYHj%Jdv0P+H>cbFUvN+Hct#X1FdXE5I&xfeM#Wz%x@F%jeL^A~BSI~~`SdGBtH0Puv%)HJ_E4>hGo1w<!4#b}{OP1F*wBEB<V?qoCZU^U_Ey@qnXvWdc&g@#f-OexR(MRu7TFNIWREJt#3Z8ci$^1rTE!^e4z^$(R(J`0uz!irfrxYWz{gv(XAG~-t)&?qEp?#8s`96?fmZ}R+_1yQkrVqQq62?M~6uc5e&!GhgD<k3Q`C%LkDLTx&d6wWY5xJYXLYo9xY3?zv^*1fuNvb#sq`LDcqUI%oNG%A|ZLAlG0A8~fU??h8yQ`BxD)!>w18u0yA%vDbV4)c^qT0>lHFeMe&(n?y{c9__j4$HP3X`*P48y!^mQjW{`?v?kOnRSq{P;8t>+GKMxtPztN)=Y4z^LofRsr8}hd);GG8bJIZ`KIgCVh4UfeVOmz04yIhm)sm{NS#Tuff9DC<C2l}Gc0B@DQyiT1boBu<in-}?A*<{iBEW^EQolxHB?yQdhJWgUK5_AoQ^EqAAWsQ$KbR47ue)GX<>@A&_InNX<@o2<zL8xQ?7@z+oc6xn3edVwOdNKGZ+ko4W!8uAo<2~ZSHwaubB=KS`t2#vp6;}Mfol|t^Ny}pBxf5xuII!z(WE<Tvp_0#pt03PQ2icj=Nz>l^nJh-iU4ylr|4kAY);{Hm+Zm&#A)w;uIHIY$fA)%PSt5A^JvdG8n=tnkqg#)Fr1+)LxBF?ak5Ep*n16d;*S^=t;y)Ng}h8KvGH5{%j7N%wvmQFy_g#>7FecT1|{?Yvj;>!w@;>Wp*g0-etQBXINE(5cq;&L`%2g1SO)am<)?dfeTq=2{;W}{eMrQV9q~i*Bv}}l@5D64KQy6t0KrR^$|4NZ1B{`S+f>^IH&-iEf96p*T`O_o>}L!VKaSpr6oSLtVktJAIx7s#ie2N0D3C0Mcx!{yc@<gNjC-8kjvkbo084|9gBQzxSg>9jZ!F$tn(68Cfc-9x>%nN5xNG`sAbtCHk^OuGL5K+HDJm5^A^4~UDe<K{f*UN2LG*Wd3()79*xnD{hgW*?cdWB+%&E^)Fpl|_{7&=tT_*NQo#JMV$TdEkKYlF<2WMjN4w8>Y!BISwBs})vJpdytPWa?V1PYCMz=ZKXo}UanNV03TupVs2EF389FlWdl{hOamI93WPx}FfrG9(l@UtIShB%mP<tu2x07KZe=4g2(zd)5ikP%Jlq?iIoAgdRDL9%*;H5W1JeZJR2^mPeS64BRjl^jIh=}jI{7w9>b&I`BX$w^IKh8;jOdEn{bApbt`(_=bMGGK@Hta1`wuepGH6WzaONRT07l^r3}`d8ytjP!Z^)A%IKf}>v*eB1Q<V@O2`{=M`-rmN?Ic0~m-EI^0ynFo0)>1j7-QUUEYV;5GTklpEscl&-mc^^y;J$F#u(^mTo_qkg1WbKN~x^OevC)mw!ciHDIOXDgCv&=V}hPsCO3<)G)CSbBV7f6`fPPVGU8AJ_!$Qyd+FJ#!8yBx!J%6$#QCSnP55ON36Y}4WxrBUI8c>64K%Ic=bm3yIw4kNt&IQL3?1HbkqaT82$!n0taP)gwvwm)C8ff`%|>AK(;6ce2Dil?|=s2|1awqm)7#b9;C3YM^0+YGUUv*SVedHYXAo>Y-<xaE=JwXh5Qzd&<dC1mYR@+Qz}6fY~d$Gwx`p#8tVJ8SN;=o33#(tX~4EpzM{Osb^p-BZ&k1Je{#Z#uw$)kwMtx@}hYaDCx12G6)hh<)iMQlw=u8LQ}Bkfe#;1#`ImP=sp8kNJ!K(np(Mu;2)BlF6F5`z7i$Ka)W*G3U$w70M@WdBZO$R`WC!Y<cnk2;W)(9PH*AkNY9<q-(6nSQIDL69Ta{N}$v$SQ;K$jZA_pq~1ful_iWT9l#)d)!aDbO1)Qz+pDwWikLq2tIZ|75+)5w>M>S64J=%)MjtMK9CG_G6Pm5m9=q#`bY*;qq&yG69M|i#@(lJ-sCv*2d;1<-W~&JQkWMc@k#};s9m2Z&MDP&njVYP!x(tvQ;BWz%SVqmy8bf7ZA?uMaY!OyOo^lsSbqLn5h<}2IA_bOar|bj_G#OH0WkQ{%|4X4M>etgY-57}Q_Q*d&Pku#=CDtQr^v49xk1s^6J={P!K2~HX?C?~O#L-OjXL;Nlb65SFLBvf#L|NLeB#X7j+M9z;o7++L2|~T*oDkW_EdU`w{fekr)B6nk+f*79)H-fMsDv7r*{y3fL6Z)z`3VPuANMrlMN~z%UR_Zv3>qL#%*MBv5>vBr!<S<o{w0hy65`-8kkHj(cB5qL#xcBo%J9y4{^HA+cwTk@Aey#u=iw6X895{32I$BO42>BY8fjmbS`HdeBW;3z6;tdH4jFvv)t`p@ICKoDmUz%hF=s{#qUhMCd?IKW3!MRLZe!dOdA5AAhx5@6FE7%Gbf%y3+d7IKrz+n!2#Nhsg>$U(1WGv2>(z<uK~hv3^k9b&sggIo5rmTicAE5dE*RU*Ky?RU&wL#oj7&<J*)%r@RxlpAB@+fX)$q>q927ng)CNUQWTZ<|Bd?q0*&1$g)6?51GL29kt*%TXPw?Ps$_4GRP{>@sZK>%|Rde(ffpWoC9^pr7$nGOx7c&RoAmy>(ZF8_?-3VhZ+_Rj+(&7$6Gmw@|r|kk{>&EbO0({+d9}%M>9iNd*@6BTUu3Hg&TbLPP%D<A)I3gpDo!q=E+ccl?Ivkv2*#uJijmcx#V-+0hQ@~Bk3IOb<=s)}7kC>ynJ=wj8_qA?BP)Czcf#X%@f7aJjRT?7zOFB8>Lp*5mxnN@3gL&k1F$jz#<n$JCKFr9fwGgEZGP+_aUgEx;Pd<_6;u+Q|J;cEh5KheCrr`W74S>5>?C|>+50{ps^n6Q@3e)p#yXAis-ImRp^O(a}vAG3i4{Kl@uEykESu)M#C+&)XdX)19bW{RQ$!9XxjEtn(sOH^+(ue3np?PY++|r)2IqYk1zPLPPK>A_eo|tSU3KD|pcFR@2(VU?{GbpnH**i|D&hp*ujVFl%6(INuxxVji^PZo8L4S+c6=x0J=pFqF;>g(eH5?&d8d`gB*!D$3o175X6nur9JdLJp5}veSuzTubzB@wwG#rDgYV5Z>=s;2rP8F$)J%&~)6}O87YKR2HXfcrfp9y{rg9rV(xWKW7&WDB|MpsL+_XgX-uS%2h+7;&~y8)Tvgn@jV5|Ny9ZX5U_f5Yq~NJ&mIc#v69jv0cyOjrWLbQg%0ue3U%v+(`MOY%{asf*ZOb`wSUzGpWvh0n$JB1K*3CxMM9lr9CL=FRnJj{`1Y0T6rpKf20>)iNI`_jSyjB#IOW7go&XDyJj^!CfQfM&W8x-C@<?opmkWX|X}jEcs|IQZ96MxM`=n0hnHQ6IcIGV`?P#<dE~&E`0jwcP>Hhq2aEyeFV{aOR_OgsO_N0h|8<Ioh*P^A&BM%(8X-?YFp;@XKhAsB{WXrc=+J9YXXtsICX8Ml2}<A)5jX50G^v}z$M1!`wLGbQ<ldpj%aPB$ZS7FzlwTWNz$~6;^7)Iq7&4Qc`w{x3D6${i57`qTniKd26)bbO#0m&cV-hu#?7L=c!#)8xox;jWjBjy;gEwA{WYBIUftq<#2HXkyl<xyu0I_tHRRh>_Bot2>NBh-%NL`7@rX;6Iy63-usFAYsduoLzFbPH<rnV>>)*DZfGh=&O<+rVT=}P%RR};4vr74>UYHHfO6&CA9t4M6ldlCd=s@A(((^gS)73qU@T#c~4;6?~W#FS)6NF8g%K~=u!vgoC#A{YA(X7h#v*M8^jP-UTx#XMVqJzV2d}Z3h(B~4go1i_eC;ST@ywW1u7=O#C0BDEoSnL#O5+;NA$XPRX)t0$zdzcdBJsFFMLs0t*FPiiOu}&7NuQ3*KeajWcQ-))bze4{iShnYKbpx_*036f>%UO_)MDTd+YV)0h+jOEN7&FX=e>7t<EUytG<<oFk%rZYHvR{cfLNHay>R1w{u|14O+-XF(c!GzXLL4&uz&u;f+neMWt-KR$^%asGvJ->S6iN<lzbpos=Q`^%q-1A=vt2-+5*Ekf0$*=<ojpYIX<RQ(krb|Ja@lE$Gen#Wy2`$Zhj1;Cc<TP_!B!&i7jOW>KZ6P6{+6`Fk)A6`G+tRJIjIBtT-vqFSM<sya>ULKQb3VgStzC42_r4I(j?=udj(Ds_Ld7sRtoAgVjPON(pF$tt3~@u3^p`z_CQ{){DP`UKKB}WOm((UBq(1l7x;`&a+)i;?7APlg@nr=3IaUEx<n#J>^s)EfMhiK+aNB5XCLKnh@qPFL4=lwGF%K`Wt*~rKGNyhUq6X#Zbna=`2<o8)rScy!-;G-<#>}L0Wx)ts*Sh3B{=MUs!u9hAhWwMe5_f$LTZZP^Vn;~@J-Naov>yCio#KaBj{q4fP{02#Mic1oQ@Knl8CYimRgmPb0K&A9`gAy_siF%Er}G^@3XXXE?mP+6jh|jz?CSZG9g4^##m5&1ynHg@wV>4mFbu14mx(Ax9v$i?=nXGh*fE3l5C?qCZfm^iiz4_#R{>6GyS-p2s8sUw6g3s#deSgRpA!)I?!{IyKn7R+-!bj4<+W}(m)tG%)1f}SVe(QP&A7(ZiZl#Nx8QOErlW1TaR0jq&*vEY_ikxu`+}d^Ad=wCeN<r$&bY_+YTG7=11;BpAsM_`LAg*j0Gg0Gxj7)ZnT|jL4o{c&WRJoQ!#O$A%iFYL5bOTz#dyr3z6q&naSDSj>67OBPcV<4G==5)Cfgpr)E95CH;&Ml+qoh$Ht|{9eo6V;~&_GyIm&OjEUP}p6CCDd*{`^$DLJ;nxu%}j}H9hRI|5(XJUH{-bh|RO-@%c+xoo|glz!YXV`l1ahua)A=zKqFe$QOrc_iOZWwV4VqQtZ7H(24y{RYrnF}ht*S#wB=&id`eeSk0^lvlVunHE#oR>ug^sA!xD$mYGyBPX*<(XuUPH^*-4%cz7@hhhMQAu3OJtZ9_g(?`$>t(u?M*zuF5p44P*g$JGm?6OUqYmvW20O|RXQ<skW9`oycZxzyU|h()$LEPnT6b@>@*Nnk@pC>rced*k2$y(s&MpwL!%jl0%ZW%@0xim@z$ILPx|s6jmrXDY73iidiUT&|J6Q{brqMdLa_pcYjO;pRt(RNfRPxzPK-1~#&zXBxc%4UP+seH&cw8!uk!42PwF}}{M@E9$_L{VgTc>1svIHPs=N}iA7C~@yF}`FfH^=E5Yk0U|4Z~?iW4#|C4GXCF3kEv^;|tv<%1GoAlx6rFdjkVSmv9IepoQrfI7NyM0d>7=mSN<>D#WXM{&+;SkX`izDjDgS^hBKSDI!EzV%xbEZopSb@3y)9I)c&q!N<DnlmRjdP8v^;c)^Kt-KuCvZOxSwg)X;OcZ@xolv!uNUuuTz;TCyfST)duJ=)#8Lfo0%AY(rMl<pfgk=_$TTMP=f2%^^=thyk{8xXg|b)KK~1nOpRhgS;Jkv`q$g`8QHS@TfB{EN6Q3YBqrA&vAi%vJ{}3ASa3G9#FrOf1aTn171$lP1veb*N7o$O`Kgf^+63Mj|yvqjC$wm&EMAtW%1tQ?m3y4|Po0Ip(EPS&8|DWzFcksuRhCFeYk`0)?<7VBf`Yh5M{5h1ORcr6%1P57~u;LHOy9H=kuZ1me)F!9s<sN8^5m=-!b0B+{^Q4144lF6qiW>65zP;wiFAw)rhwsL1L}Vs%P#<+M&7x9(6W?va6Hog7i5MO#oxzXC8@O5<@~knt{d`HmD7s~zlVmtXK}MAl)~ILSJurO<PEQ4VvY>;EvC4*|EW!#=M5e=8!oA=`XM=cJZ;-6ZAVlke*aI@jx0$@k>2>)^n0pz0%KhHRx4Y9N9fZecIBlEM;(I7v-`mG*@g*%U^DOG402feoyRk`97_k@6!(^mEVMMrHI;eWYB%G`T-XcK>o=pw9=!JMt=lt0_SYq!x0@U}FzCgu=jQK7^%x9f(%D#a@~h&aUGb?<fe&!vNB2ASEWr*Dq?l?sjgfEwBx|=qY@SaI8BBOb?x23J)s#8Fue~tVnXVlDJzqCEB<ih$L*G^;GJ+HMK(-%O&tv+$5riBl#^|k)|kdPBqO`3RhSzB<Tl|pT(F{N&PBU^>Eu1UTdV8q0th&fQ#H0>?jrt7h_L1Jf0Bv`Qrj{aZ6_4{O<d>y0YCx!?|&^P)V1Es1#oJ?>tCO$QZK8>wZ37qYFBsrVE_uFFy2T&tR~)xNfjsN;0x?dfR-QeXq}~kQCIM%&c?<ewbQSoX8XoPdIMVb2fRT+X%R0zGFd=^KBQVAxqB!HhOv}Fy#u`t41eInkP?mL!SNcL={wI($uN|)Z*rU?R!TXEE&Om!*{fK$<1&N$lT8Jy36=sD$b$%R!-5Y<@ss_n9@JtgTIAV<zKXl*&cXXjcFI@Zf$tbWN{8Pf@lFRDcn8X*_3;$%23R5Qrza^&sef5ae-~D89S8+kQG!nAE>V5yc;2ABSX$O9I|SMD4Xq?jppcQ8g^VMMg<xm!DC-;C}I`YN7PSq&J?w|90R@Y7~WYBNCh;gOJ8WWf6nv6{6bDa0$*GDYmAoQ!v1yGRTw}!PV}Lc$8a4+*<l9|s6wYXo*`6W++D?vZ6V4fDIar&eo6Ub-3cz$drrF|lhQbYcWO&>luQC)TM<UG!13K`<yW|LGtBQg#7e@BXvp;jygZKg3c4``Sd_o%&fVl_4}NRAK>pC(SwLFpVco7&(FTtDeE3g`gUYaHDh_@;UgR2^IS(Plq>-`qh?uTww=#@!lCU_LXM2*FC<XAFmcg0#v9X?D2&!<P$Hm<j6cp7xBnTjuDP9y-Xb^o#f<U|K6iu#sP<{nlU&*f>>t3K+VOK*xll!$2RYEz_UdEGmeJKJQl-u-^cc2?!wQHd}!vM>->{7rBMPsfx4F$+~mr2vTL{UIEoCsgSrF-YUbj>Z_ClU0FK{87UU~bHQg`lxTkJ9CHMVL-J-6}=1Rufo2O!czTjiVjumLqR==EuNg5%139q1ja!J&~@QpBe;UzlezE7hC;LtPwBrLBNyP=={{kJ#=u^s9<V>nIPv*2;U<#!>2iKaYNV~7e1ev!5eMfj!~JTcDiynWNb^Duy!Rje_8DNMdof&Olr1cwsGKSZOs(fAUDsw+gAdqV<|hmMciIj<YX<74Evp|X)p;o458tGDr*W>T=d3^EnD-5Boe=uKxxC%_+O`#k)n6F0+lE6QNyxync((k*j)Hrp)#E`UB@)kuO#9zV-=UYanBaX@z`0~O~%H(n)$QONC0z|rmso22J4E2N=}8Y;m#elKn<aJ|47&_0)$9{cQ1ha_`-8=3Y?#eUyuiNB@Jl9>SLMLLki4#-4q%EPHlIQ{^Ii$C0{zPAiBlLqSKxUCyCAuZ|e0nApVGQshbO!ZJ&FYW_S7sDM~K7!4{+_u0Cm~6}IR(U(by<ITVlbuy759?3B!GbxcWzGrB|aJ>9O)$MJ@1Bn2g{Czt}cCt^I>2#EXD-9nStkICHyN`Tyj1*^^WXN0e24uMi+uR<f(uZp+{Qb1>b6+6GR!OzKRhn}AjSS^m47<;m@?jurwfj$YSn6t6TK(N#J8*OjblfX<#1DN>TAm;ZX7M3tg^Pc4+o*bjpAjmW@CGOPyH^Cw@v4*&a7W1=}N0Hb~l!&lc7eCd*Tn`J4{tLF)2!9nF7^XRth44biLZ>XcP)UnCqx^)Y2^M{)HMXg4x@Nh~(70pDeq_<I2sL8&4T~g3T7+`wFmze+)Zv$-R_%cghb0<3tNSzcqrwn@OylM&PkK&O_jZ{YH#JfTE%uWvAbesPdq7z$B?Gc`hU!p=Ky|Vifleb5uk5FE9SLbz{9@C&)+4yq{h@{x3Hl|B*#9fG#n_6B$N@O1n2e4v-zEd_VBQYU#FCCfdTNLX)E|Dq3}-_DV*_;k8{6PVX)6xs<}3G+D9DUwRf@WwuL5C*CrnoXHgcbUdZzJsM5x((c1P9ju*ogzoA0Nx#6n%Kq&#6n7R?IRh11Q29j;4A#wkEERkpY*B6oEnOf)of8JgFM-d%>m8oGg)O-eew^MDK5l_UrAX1#<o*BA#^4Y&aGRB8OX3+VR4JS#y|ta27=t~+r;1_b}If&%-BYG1dOA<tI{Ze>J*8)J>ZbhI6`;%-hp6$6lAPu!=_6+E`;aU=s*Nxh!v9EiD1RPk|`Sx{ImYwh1C(g<>dwO^Vyr*BoN*yaFQd4gfuO$g4$B6L;(d1moqusfph8pl|foEQkq>9PVKD4bQA&l{sK?vZR+pF~mE9IDB*Ix3H+uWTC9shI{}w?AwJ&adFR!g&Olj9vFkqB0Ah2368zGR^>`{nqnim*^s8Bsaq`a?=c*kgqeAkm#)>>##brV=KL;jq0C4H_(CBt2fh*v^F8FUs#kK)&vNxS`ii%1h9e;(=}>S0A|9jy*VJtlv%J^U60dhdi3H!<RJ~xHLE1QS?_T9ox=~B%kLbYn8OuEubp(UUM-QHkbU(jeLqMtO5ANsN*NM}WKL#0c-ucOVmoAyXVR{{MAfkSS@9r_0J*}%pXyBxpHC(smlXt9bkbOEHPN0UEsGS(Q{TJb%@l@&6ox8u)vsL2d`U)JR=zK^V=%txl02?Q@+kZ+V40R)c31=@Z7+R%oK=1LIRnLxfR*ai?+!NpkmpcA#mgL(gc)_xh#%zujvw=n7*gWU^tMHjI+XZsf_s@DFh-^R71ghh9tYCs&TBPvxxzEjBZaEB=t5a8{cVF!=0A;!66+KoC`6C?BkeYR#VtM2ddLsN0Q&Fv(EO3Gl4%@9-ju>;U;75YT;9~VDuIWQMe=d%0bMkyY46WQFAR@32qzh=hgCz$8oU_jQwi@baqx}1Uc-Ez6!H>xR=b`dkvcD5mM!j#8xt=X5P>Y-LaRiTJ*We$BVN1`Aw>MU$)|9NKy~F2)h7{|iwGt;yBjG$>Uno6+pqFDOnpQq3$=5J!0&M4?4IV(#GTfDamY!fBmy*)Qi<wOwpw-;$u-KXUzZT!wdkz9wJmsIS>hvXDAEtCUO(ZEEi$ZC<n(2^Jq1`|<C*NN+jLRm;xrBV^y5P?h6V?nIGY&%Xrcf%r_cs?_h}}pgyR|pn;Hrls5Yshx1vf^C0AaHfDd;g4G7T#>k%O!owXMpo|5T<DI~VKLfLR|EY8t4vN9r#(VaT3B{Aauy1HVl3{UX8({r9$3ZK|&iydc@nhTnk3~$5l%oIBURyZtr@#G$n!XGKrWMy()14y&Qq`f}$Wrn91GP&75Th%Gt2t~2vdkJQ?Y!CaR$XQ7K$vM0-h2R@pRg`ECiX%0u=gPSW0|j#Or3faO9)p1@oSRlEJKhz!=#FJdon)!IJCp@TkWmjNXw&aKy^s9a-#Je9Tzg31d{Y=ft8h$fEq4=lsc3l8zY@!hv_8T*dR{-MF+TkncfJXRZd6xL`#ImPftFOG_R|;?V)nuq5T9wz@T2B!&r5;hE1PvH1q5>lR_GA<J^^wpwwyDj5^1^Ee+Q?e@T|^J5!V^B$)O}4%**Cq{~7QrG&^t$hx{7?38v#q0il8r8)Dx~4B-|OL~Y$<L!8&{PM=AwZq~$D$vs7avPpgb22*qjkTs*%PX%dW{mJ-F0UF3;2(!VVPL~FZBk(@i{F@>&QJ{-S7MUTQTCCh3$cRV)0jk%p7x<7Lpz#fojw;Z!C0(|*45+CJ1Lg{>+r8AwwQluhbJ{3d<}@~?s6<_C-WuNi1CxpCp-UZ>HN=cL@j9J*97!_I9#3*hupEY3<d)!BjZQ{~&-r=`%-ehbviYMevwDr^p|OF*4-;x==RO0-4L=+rLT4|Dl}o&ypwy%?o5negq3bGJ&VFW))ByeYUS$&h18f-_VY<<vn6?VL>yenLYGeCy96C*tiYpdWGf>7ZM$vrCW&s?q*>ENajjCXKwManl^hnmtE}E|-8+9~&j%W>Bp6xq6-71y-tRWK1;hDfKkD0`JkUIib24alIDpekghSrgVC%4@#PF<F(W`SUPg-M`H|B^eEdU;b|DdPWsSQ9Q3j@M6TJn5&RO|)P>y@^e3ZbcYvT(zaEkXaP6G|nNA47n-#9zv?d=Nrm5^j3!}!_!`u3`C}Hi4^%B<yrVjA27~vjtw2mex14CRB-qNM3E2#DV!swh<X7Yx46PbYTmD=7(sVk-~gnM;gloNWW4>M#`ds6mGT;IZ9enDtA-@ZoujIOF0ux5Q~a#;B+qBi6uQv9_sVGFMq^MKX%1tJ!a8G?MJAF_YkDV3)u|r7s|H%z8_eL7>2L_UG`%MbSwR}sGrISc&GLRC|Ew+{JjA#KudOqLLnERe<)|@%0$-GpKO3`XKDzS6HWJ0O?44|a(s)q0d{%=F&=Y8ji&rks1t7t1+Gby(KAHwm5GV3lUbbTxC}26fHl~I2X7@-R^b~UJ{5M11ZgYOBrn<s}65)u2g0j+*+1=n%xLW!0_mY%eW=uSCncISqatl%+(_l^DLZHzQ>7o2&*zU6|qOZn^q#mCx05q*KphmiLkpVRwev2ZYbQNG`7ZU|L1ee-C9KQ<h#l0V$jLTH$uDo5u_3DDY2xv?NwBV2=0ZVZ|*DyU9%C$saL_0z=5w$sBXe*3g>Dr3yr2i!OQ6r-4VM9ce_b|KB_4)f++wJIiWkB-XpA#$HZ5|x9f~%xTi<H;3hiAp=Wy-$9k)bQQfyf?L_8s|^a^o^969Gz;hW*_M90eObO@DLvGjL>$NpQpsn;3UB1(F!+mkSA?eSLhw@>Y&@#zm0A*H&{~ESPrn^zGihzM4&glxe83t|>cHSR7?`*-sa-lIh7l1S<C)mOVC9aIEW0*CUrS;IB010IbN_jtzatPw;G4Wv=6?Fly4E%`R5Tcun*yWK9MP*ngw*>UI<08|IXSTzB{D=x^SdtLC!9*fXZ(Rb@3zft@j5F*+|d7Go)XJ;CnN)fV=OHp?UB-uYh1uAK3`V&_4H4v8(%&3t7W>EyE<!~%+rZS*GsjJ+VPEMGJJdNrD8K@^v%kDIaK5!8W3dW61*#272cIf`B9BP@nXB6T`D#^o(SjShH3<b?NRMlT(Yu^P!BOp0x~7PC}G&DrZ($AlT&K*xJj_luwk9VBI^OPjb<XcgVhq#A_x=_%eQ(191g>D#Y^PIU`n5_F(<6Kx+UT-go~QgYl4O-x8!z3R|1ly1%MvoD5@TNP*OeH}1cNsPiL(GVgcwz<=#;Z>*?)eYdGBvqE<4za+tAs3~4#G@0S^Q&UWZWo=yDyQktTF9lkh*@7EI#<sF?ZMrNgrwvqjPtRG$)OD^O7FGkqt5Famj)S)NQ_VYHOdM|-uSM9<W0ybs)&b~vq~U!G3+mkyF!$Q$XJsAdwcF_hhtIrcl)e_BJAnTLnUa1BnjEyN*lq~sMyWihz}!-m3&snBblexofFYTVWn?^H@EZN0J4n)8ZL2&{Xte(e$*Kj^lmoK-O3!7W;x>XC)C~buj?r-F}spED$T^upuTvw25cDze?)Q36>wAK=o~j1$$9Q*UmxP&Y#OR<WW$svew!G#ub(dXNLKmCHHbn7D)cULQu7Jh2<+Ms3NJZ`aD(hR(lb<>*Dv;oouO-0m)OOCJ>}V_OQ9<@a@{aB5QT2maB`5#$_fb#qh=6CBeDQk1nhny#J&V`U@MX+yB_!6#XO6=txR9;MjlUu>~11OkUS)od`@cRNYO{YOFCQ6nsg@zL#M>rg3h)Is%ns)lza1%&gJ6@X>K*0>oL1fBBi@rdYk|P)5+>`jL;t$B-4%d;;t~^e?ST-wY*khZWWJE><$)>C^3Dl!fBvEu=|HZ=y$6f&H2kLAiV2HK5+&CCtlq9?ICdYh?8Y8gqOq&(Ei9r5iLtc0O{4QdpY^PW~C#8At%bP#k8#v7GBnK=&;7!AL9*VL4<R4g$O9$7<;)IVq~hrnrUuHSuC=<@M?cN>YfXkT}h?y79(TUU!P3a@v{am?h-g6y_ixkESh``h6!Z5PrwV6?IftoodW23Vz;5tF-S4wdZNUp{+z+SKkS1}M;M{e>ztB#JT{+->QZ+-f!D|Ur|XG#t-d9u>v(-$xd4-w8|+n-y;gB;l5jIa<$;b&T{CzAowt`udY+`@72Pd?c=H?dD-6aO^4c#LwbeMF)|0--;~%F@;*nnvJeahMnFK3G%@y^1x#TZ_PENLv3R*(Q5+)2o5x2{EVJ(_$9%!!bSb_#JfFjEKm(y)l&<Z$#b5|4gtq3EYWItKxlld6&tjh2oFZoSHsgX$*W&s{@<Znj6=Vf8Oo}4Re+H$yX^$7pq0Mss<Yg(gOKdMF!Ky7=*wNVhcwzMrcoc04q?dW#PpA>HT!9X?8SfRq%-QIm$fDaT-Zqt1i+G9+T0*hJH9(dk2T{4|qpKsoCkx@p&BaRTl(q2>dC9mcicFFuKJ7nk)GcU=&FWEmD8w^|Jf8$xgKJKv{m@eAdRJQo>a1Gj(7nC34BM)H>^Rh*+H=B2=W`{E(!@>}^xG2NvUbG}h@&VJmJe-UPIK7N`&)A|Up`>D0wegMXGj08odf44Wb%^CYuI60s!jR|{@evz#vse2DQWtg2e7`sJOBV<jgffeALS=t)%9jW@z@F>PLmjfkH6UWDtZ;isKV6-WAc^CKJEYIbnQCId-T!_5Qz_Y}QIgZYSdo_}3jq`}Nv|f5&g(TpBL`UO)+@JS6^_vp#~_Ywria%`u?#fcAq;BBbRyrcJ5*hQckO%DXuMOzPL~x6W}u{#m1;*X%`o%oF*`$@j@*w-;%D8?b+zY`F|wqOA@a3h$d~1eeX$<{Ok&!I-ry|SN)$dA=gYO5UA4I{VDm_f#@?Z*+*mvu|Jgm*iN1M+0Vl?ADT*?GxbV*5{0Sy>xJ^+K?)h*-W&VOw1s~q>=heegikfSi@+%VByaC}xjb~?C2%7C}@I{NJ_Wq<EmGTVgcn))&b>kGu6?My{4=3inTT;<Ywm>$~M2Jl|G=S)jV|x?<`$^$QH&45X#omC9kQiZ9D$!a@pC+pT&sGSGz2If>R_*T<J0HRe&HWPZ#3AB24zl)!$ak5J(6*nN<?arrVKkB-9#GMt_+i<=2biOX=FF@=LQ#7&^u=)4l@uZxK=;Q0I)%!H?OFAR$PtrKHr#k-<2mU`!h(-F4`U{^?r@eDAHq^9TG#<^D8757DJn4tK^{Y#x#}2X3wnfOLO!c_LlcM4MKGf^zz5JKbkXj^xiz%Uzj>ZQ;!U;Ih>&<J0Lx1>M&fgOXT<Un<_1|+B<#o6LkYTJu`B$h{5rM?24&LLao>RX`H(QkSHZEo|J+d^ZjqxXFk#D8idjWs0r1FOATjrNLDCXfog(?Pr!2B)p(vck%<zcHsv<k2)Yo?PTRCU$ZDQ)Edb4JTSIowDX{w&XnlqvD7}Ajpf+ThtbGqZ#O1r}W9)C_X*Sf<zpgo4o<WIQI;QoXnMpRpj$i(3UYg<E7e0OenW$J_zxq|jgL&LEb;zWA5J&-FnM0b<_kTIzi4>GjkLss}Eo0M_0XG;3Kz%ab+65wAy5vsGyM~pJxtYgEgzxlXnNLB3$>mI@+sqr(WxHG1>G|v3TcxMM8NYD#J8${W0=!k`ZK2xL~YhtqofEg^b6ec_SS!T4Th12hrFAv7On#dt%LGc10u}Q;)k%JoX&MR<JJ2A!h2A4ehR`P+|Bco<3Vp2D%CE6_#7Oae_hMp}wp&T?y?ynGZM#ThO7k!@G7Kid;_`s2F2I&;5r{#kmwiyir;5+&GlIn}xhixZ*ltmqmur*44q;?I1Bg~?^4Xxi)ur}G^ft&rYj!}%*HKYNASTv-mprb{OJm*EP!Rs9(lTlM2N8bzEXSz`-ZL~dH$;hHk(*An@@(h@jpvQDCiT${c-2ybaJ8^w(uzdL6q$eo^E<;MVFrML5O@LOfz3FxN21wf^#n+=bd?w2Kjm>K1X!=Sa*iQ<<RD=uyGe>wVqk7SeCL7)Wf^37eFkl9adnlL?mjOy4C!5;Wgi)9qj6vknFAmb93%|iw(pJ|<efom$VW-PKhv6=HMbkZ0#{)6V$VFl<Vcxffu1<Jgk2%!lC3c~`>>O}9!!1OpF}-d(@Km1!sU(g`<hvN7YU`~Tz?fBV6}K24I+5CbinlT351M^u94g_YtDs=+iDJ9isLkJwPmJUA&Vh5($Z~x9bZa4-6m?EFX%E)agMP`|(%{n}4;4c)cv}LyNSAsNdU9N&vgkGcy@)ihHIes=DOHBTE<Calan_OFSfBhx`$P^1{s|go6`2(@YD^rzG|%P7IDrj8T9;R^I&|JpQ<N#*44z59IRt1xcCJB=c+c+b%-h{%^+Is$_?1Zv9GoTg_7Wm1I=H!O^LXszuHK{ZPIrG(ImjyWVe>ehNnUWY0_JqiZgnM8iWpko92<Wu!__nWG2>P?w8xKR#Zne%0<z5D+>N_VnO-8g)-Gy4LSQg#a83%ZZbZlX92N!w=JfO<hzg3EHH6Ad=wR&CNPw1M?F<ztDhKjZ0UXe!(2b=y21DiQ?-UKL20L#5Bp)tNDWwQDW9Tg|J2d-MoP_zf(U@u}q5wd9Ewo*>u2{m2LP%kWvD4Y4LBd4A*FM+QzFr)5%%`n57+_8%qIj*OH&+N5<98;+(iq>a7$L2>B6!pFq<`#9R!q`JTPuoVyFY!+)Y2PJ*sq}{&b0oBaar&gUP~u>H}zwRC^Vgs4zxU5D~=8_p*<C*J<M%|?>Qkeu+b937xjcKaJ0vRL9uF}CNjI_{aHjW#$A}rWW@QJFeHc8z@CWqCA4L(525FX{nsXf10cTDm9Y$$9P&n_1SyfJ+~_#MFhu+6sz14Piew`dSupbrUMC#dCD^p~(%-9sKx$BK3ooaQleR@98>J_(`A*bmE2EC35=~?XizkeJ$oW8gG*SSVdu$g11KMHz?<WmQkLKRF&h;pseJQR-!S&k$Ww!gbazi4@Ay+X?GkE_^HE4$0tB0d}_mEWv*@h3@M0!e;A?+St#tPS&M5b5^Ey#5pp|A|MVtJCPJG}@p+!K-zT9Io_1UGgh-sL-YMPu=}rqk#i4%8$*3;<avPCt?$&?=FIf~k>MlN<ef3-TM;DVVLXLPF{o)wjmvW^e4sBEJjvA1y9*V@1-+2qmtEGN&gwgePhRAD!kH+s!dr?7H?kpHMf13a3BmGc%&zhG%oF$0M`<33cM=)B5Z2kf!;*TYvcA`J?QuZ>Dj%j$ma&9Wf*Hfqg8(pGX_9HL(d-=LDxPR!1&*>(Q0TpVXpSB#(^TmKt^gUDLM-dR_|i`SnuozEaXO1i-8yq6DX_P}WmX<}21v_z@m*AX-L2<?siJT#-2tr>DGm+h<wccjuI3?CL&`Xe-?ex-H=micBMzfP5GZ|6IV8r0_jItuK|G$7Ab!^s_YWNA}Vc+K`i?v5^wP4ApXN@202A7RF&`avb4(TIfQ8p&hC83@#gbflYLq33+METRGkF;a#YE-Kl^E8*~|R_RVk)ubVd@dzC~$v%k{1AwTM>3ek|guJ{iq!K&u116Bb(Z$)Gt#fzTmOmx!a@=GJu3>gzgtO@slQ$S#2w}bl_P)1Buv2=(a=f~rz<Fsla50$G+0jH)seYKye9k`Cya2;q|d0cl^d>N*+f;K7Jy@|^#jvG1%gL^K>-hEpr-!brYi=ePm1pw<BBnz_S`2i^+3cIg+*3O12WLQd#6LWIeE!-jnpwmo3=*_?Y*+H=e_S%Yk@cJa6%qLAoieo1L7iDNRcmA*PP?fJ8ypP2&0lY5$WAwN46|^j|i#A90?&2>lg^A))@QxPzUrOXiEl%le#)M&h3t1dlMDAWvWmsbTdNPOHmSoVdWVjTr?C(q#176;~mTPUfe$BCi4n(m=jw|PE@BpQ5z;0)`oiii3$FuTY156Wz5t^`_6d`vLVk%bhDGVAjyA`JwIeyT}?c7+hjcW~De~#cGTTSJ%L@wEh(K?9M4u&8>{S+L9=0M;bqw>B&4Zy(`8dl}mc%s}4&p|R0Nm5?YpmJ9UsY*dUE3XH-?t0VJ$UV<X0zvv_s8r>0W@Tu&l$rRc+%Y`&lH+&EE|~!kr2zU#f}bA_Nl{73siEMEZ_zWh$9a-vDr`;_IaHs>p$&gI-*D~0JREOI;h`sq4Gf_rv4Mlhyue=*@5|c$rd(2&8El%&jR!P>hnCM^!qB8^VnE)N=Nl&F8+yS`G|4OW6%xPvxfb$lU0i)l&X~ELXj1F|Vp3jifqPuI4)m&l!EhtvO9P(Lz=OpbJ%Y!CWih$K?AC8`t6I%1j#oE`F{^*Mk(Q@NWffs>m0|^B|Em<1(yu;BKd`+y4a%5crm(uDG!BlOynnJ&hJANrXea_J9O+NqqPg)j=0b~($${77!fBehCc3)UY2}$h&Y=D|HRla^_WL3EJf}+xs4B~y&pE_PAnWC#RG;FhYBUZ8pu2A-cyVy%G6VU(#>1m!vTJ|LPk;ko2C^I>aBuOj!T^WO5X8r?0T<z16vs01-&eH;6*u!fje<2JTHBcJ8t@_oTGTY;i)e_h3#>iX!|B_swu?57kQv98ER#JH>oLTdKi0tf-dLWL+yK+yz#9>S6#YY?$M22rS2<Knm#cyjzlbr=&~??&`)b<4Jlho7-9>yNVN^B3bhAoRx^cM-1&L4v={`<LTGIv`swg=`j)u;=NZ}-0-PDm%>Ewy@^c6<cJrwN4Nc#NbL}ca9Wl$P(U)tBO{21?u@rpn*jbfvAGIuAx=n58Jit!h*YEG<Wc7F2;GP>XyXZ5^TwekvZ1CGbiT)4II<s+Kf3<Xwu_#Cty_olc%fOg~<U;wP44l4Mr!##V%sz(7V_ZS}$kD@ec(DAC6L`HX`hCs=4@Q}*Vu{=bLf}^d5BMaLW2AlLCZS==qO|E;MQtX%5!ZA03N<|4R27`|20f~F%VS_GGpqz(`6y*?EatLMbN{KlG_1CdzD_hWNm}&mZ{*8I)9~DocJt>|0i4c-u?80KWa1FWMu>B-B%q+MSZ<z<XVz1JZI<QsT$SyDjo+YL`<H8z~>=F!;!|2tDxG5o<LHA2klbs#OaM9;@M4x6(RQo-+dz@xsWSWeEt;hn$BR_dcMRt*eoh<~!%)>N1o}5*wk-F4Kp%h%$3sXj8teC)GAluD}x8CvM(2{y7Y;JQv<~p{FsI1N7tV5*}BAH6J9}FRQ4>*HCiR*TSWO4G0hF8`j<C<zqJI9Q;(_+G2IZkM<Ic;&2h~$)^u;~Hj^lO1J6~jEi^IA-{tAx9t%W98lU!QM=dDc|DxUcNZ^p?9Ceb?F(CPjjR^-4_C0SjT$=faa>{G`W_VT*@nOXCSTFA*0Yx@&&KGgo&sZn6YuVC}i}h3==g>Ai50sd52PF*Fld`v}=l3SLBOr5f4Lw{Jwh3ek7<zIU!LBB%mC+<=T$o5_=6R-B_0Gs6E+^bBj?<bQ05-jh6LhpvbVY|II7X*-ES5jM16jwDzL?a;ME7R1lW7gW@o!xPz7v8ZVoEZN5*8Fm6ID`yEQ5ON<>&~3i4OR|QtKJPld<!8tn36JyfgqQ$iI>D|&0)#io`5E{seyL+pvm#}Xy9%r?UZ|LDiln`r-<Y23A%&}se1&1JtacH-<3Hop?5u4{du)WahUly_>D@D@W9dlN`*wj-m(A+L^xc6km^@ShUud2riZXD1OzK=Su=&y+2C;)+Fa_wJpMgsavHLalh1-%(o=ho1ws?blu}3C6f41S7>>%MO$pRZWL}W;jA3Sw>)4I;11KG!bSt*~e<jrXKJ+QNvj3Z(?*yOs;kt{!{Q_8W)#+!j6WwJx{Pv&f)Y0+{ql2R^cK<t)9w@!ZoUwUxxm+Uj5yT=ouzQV9yvOuJC88PkU@oX4Ll3S6-{rMtw`(6(#F&e23D0?I#Jz<fCqnV9;#M`xV`Y{|W&U0$tZpJmfiygHkJiKM_@3yg*F4rHRf<|rAyN>yQ7VC*eyX2!t{>+#?(T~W$Ksvs_;a?~r7#Ty}MC*=w%GdF{{kha2B~e=JW$q%ku|HAqumSb6#W^gSrf&nubtSAZ7$J`kJGLzh_p0GJ%#6xfFwsd7q@>BicGD}G!I@f=H|e&C+f-1o@;<+UiWMdfd3j=J31_Y}R3$rVX*}zxJ44;fX1b@3azTd1j39r{>dbX2>Q$K`V;K<>w|cRyuVs)aMm12lN02<5>;Cwj&`ng)-)&2x{8&|DrSdq2`>(9f47<Vey8-Z-A2j)Z#$|!YhutlvH9aDXHWNd}pMX;>{1s$^tkfoZ94S4L8ASmW+)Lx>V5U0nB|=WeL)qbe&e|UFF+v%(tE=2*-MGDBHQBKr((li`J5FBI0BJPoyuAL242YtEjI2MpyyUxT5<~7*6=kPm?DDta6~H<F3NhWOQ3u1m?rg?#O~iSv8fop}GRxq9VgBe~)W&_g;_9SO9nF|VhQ?Hj+!$S>MK@>AvdO#J;n1OZEpEZr6Tc2IVCv&&BIl$<yA9aD?J(8oav3-uck&Ftlsi1_8iKUk5b}JEx-Sv*CWZ<q+@H7wrLwmNFr$}V5y{+Snj88NESLS5ImpC%!{qD@UYVNGRGrHO>em{ya#6|C6Of4cx<Ep7nfnb$l$ishm~lB5kD+WKHm)*e>OQa;@4#lpz(#Hm=2{@PLDvcHr-;^Q*=nsP-qs8NXwVFWvMeYZc;=vu<9n~L2w-}7jjosF7LJ}mPMI$hrjoh~`4n}fDMTBcvDXdzj(ED;_2sxJ=Mx{p(PC-0ZyT<)96<vek9RyB(5B-YSM5FS;G+aHx`G)URHVpAP!Z*Xz2WCV4zWplEXr5n)mRbnx_*C_m#w$M!=XD(+-acd?VNX<9v>*9R6D6><bi>gL!-uXN+VD%iT8_<v$E22NxaJh6NA#1MeO3uM&c1tAJU^Pmt|K{cBIXm80xem^BmMxj^G%r*5%nmH(BpAq`-Gg$qFGX4>bdRNyi3>lKJOWxf;gq&)Rf!vX=B$!qXj-W48B*V-Sl7+$RQw0Iw*@0#M*i|Ix&j5eCY>H2rH=j;Q>s!enC#smD+vMH$Ng8zK(%kvl}}?Rv6WQon4uIzCqtKd*sP(RpD3)~hlLvtd?y(-0I}j223i95}>`%TiY899p9&@Qe!ShGk#gxmM5hyI0GD%JKc9#Zj4)(c*N4_-5fM?lm;SC8^zVhAywJ91>*{AykE2&f)sr97b;Y^%);eRF$;)E~bh6gXvyT`6UL72y~Gpcc&{3uB+uxl$&rj%*NXH;K5F2W-Ho^<=eJ-J<6;%CC~8aXa`(@cWt+6Vg%kRiFSZ0(w$*;Nfi_LC>*AAoN~N&XfG0cKq+IK$KIc`b4B^Mi->gopj`xNZNROdm*ON0W~vsaxI0bRvY>TSnB`8Q^1vOT6my(x@Ku#xy<rUNdbvhLscs6^o3>+mxHAzDhX8rsG@;2W9mX!4`z6=k-Wi}2NcCZNtq7~dWwLHccukrE+w@2t3YhAw@pvf6vEQF<d18)fM+)eV?&dg_(w5k>JfS(@MW@#aME1dh*&Xf)xZQ`vD=ND!+Cyq7i%yixkIq3i&bOqx6IYTXYl+DW?C5AZP9?Y7SU73y!(+!^fJ}A^f|gfUCwM99i<qUTI|C?&2VQ<`=wa`uNFM8(bZWyqHj5UA8(O9m0PN+Z3zij2QMj{cB#IF4q7nyokz%;mjAr>I<-8)wIVV;dQ@HI|Se}jc1!EYpxy0RI+MU7Ml60k?+*;nXIE%<pSnjEN^SJw?&g86%ys1V1gdkK?otZ-3CGZ*8oC+7bNpa5%Mh;1rwPe-%O;0dV%m>&9;s+u^C35NdlS^lw0n|@<eJ>Y&+rYeK9n=p;{tkPw5%MLu^P3mPmXW)A=W}s_EA8lMz5#O*8iE;X4^bip+mK11t@Dcuon2%Pr{p?HouqiE2sXC%!Yd%WJ*zR7Q|b}(Nej&ExXRP>@zXv-h$)G#&5GIE*@3w@cCaeBa@5?OSg(h}<Bpb-&`yR)j$8rkZAXxWkUlGda+IRRCn0#{4k?Txw>8E>Tb`#WYM`)%pWeMyU5)%*whvZgkAgR?A(yprR;qGJGZA^6e*+HI6I|aPKPP42*xfwK6Rkw64O<;C{7n$<C0X#7_Q=fZ<=n8Yf>?LcI6;wS=~XyJ9p9%a$mY_nO1p%+Il!b;ECu7edUQ*ngQQW%l*#OrqzAQ?;=$mrC>^9t_a!HXSHI-s?g1qmOElTcsBrT|l4U(8QtV9$DTBS4<IF1b!WVMw^qTNEBo`L$*@MClJ6CBM?gO*^=<hfVG9Xep$=Or;a`-2x=EA+HRpyTA0v{1=Bv{f;s<t00^lZ{mB`_3eL^L0d;_N}we&%aXwBw<N-n9FZ@j}+TGp9n^8~ktWod3Vif0~R>uEA}Jr8|PAmj0?tVMs9_q`y82G$nFWfW9koB&3D*kv+B04hhm)tGno|Xl$!NV7Q^QFqmUC=UzI(4N#JL+wDj%pS(dsJ<MYenk;__8-OO+5L?NXlq!0D3x&>h$}a@Qm0K@!%3`j=i{L8UR3>?%KpRtyff`W8750#TYD_3NF8wLJsiIAkCOQRq*iE6Xj4dAgJj0PYKa;WYIKx$P^ef!p21$}ej*R$&OiGwp^e6EkmZ!od^;Ocfd`HM5CQE{Jgp={Z*$%JsUzwn-T&Eo-_;_d4Z<=23cjh<K#j(C(O6u1c8IlI|o>myD=%v?q39n2tnQYXMglU1ZDZ!`7^k#UC7neiVNA|dco2;Z_xcE5mP1)d_uV5<PWI~yxA-7YNn21-#G^w(nxh=V@<(Z^Qr^(kZqjPSBfp-a-8{s6(9NC!xW-5;|oN8ls3{yWBYCy=r-6WK>^or}1vh2{Zx&+&E8pybtl{zr!X(Z_YSWA4>JRENaX08LNUI!x4(qk7O2-SCiU0{CS+l0n(3P>yBKE2to^*U`2mphfP1z}Lo_MLK>e15#SeY|(}?i|c;5Q8AHOfn|MyYr!k=PP3MXA3qelO`<nWG%9a+*HU{tRK*w3V#QPISZ;V8%p1VDoc}KUyj$#rjhn25$!Yok-&<QOtfmZ?fPf5TNyOs-tkO4VK_(F^yL?6Xr9AE#jz3}PBQ**xT)+lXR5@TVBEPLPL1+wA%~J}91dn&IzKkYmlBynScbe#o+^x34$CSYu^6Hj`l3UiBzUx3$~^N;W(0b1Y=}iZIFRTFRE{WK(W}w@cD*o$e&r}s_Y>vFY%GKAW?Q#vl~mu%_@@{phzZk02n$o$?mPA{CDlETqslnN8tK8u1fP5!P#9#Zg|xBRCKj<RBP7yNfKBEl<!6D(aftviD4j%zju0?5C-y~2+Rm2co(sg!-x$ac{Ayqk)8uV`@G|JIU3=9|O#~iVeLCvFvkmT&ybRD;(gP-^YbXLBlK($wtsHjPAa1lv=UueX|FQ-G+dx31nFsAnH)Ojn!y2eRfg&u`;##CcZzu@J8mkswef{#hvTj5#TXVQXCt6EXFkgBwU#LTvK58ApDHBUu1~;z9rn^7!@mn0xV8O`5BWn^P3WZguoy=ta=8>-19L@~!6)pAc2}}tnAP|**uyG2_wU466?Eq%52g&+!11hD#BmF^j8{L;35n|GsSHIImdL%r#;8_Mhc88hJ_bQLY2A&8Ax;4uDn`=sGZ4<#>!fkk@(R-VNZHKZa*k3b(F*V-I;HbpFo3UkZNhn*VNd-B9i9Fpe%q)Af!iLy;C7msH$*Ho&W2B*{ax?OGiDse$@@&A-<YZSPO$*-e`_>%r$TFztt41h!2FKnN1FiK-N~9un2fa04&p??Vx*<};Qe_W?2|xdt)EyLW2--W;>W{moQ|_vZRBrT3OuO2{zr%n~xm4%3jm+-Ct6rHYz&xDgnNe`-5Hs(Qo0_ef*(O}GBytubqYosymL4r<qW!xEq*n<M<i=|4?_MnK3(ymPI&QW(!iad@p&;oi-b-?NZfTz}J*N=yp#k|q^|i*E%gl~8;R8@M3C)FRa|Ch8^(P|aO9*MF7#J2PEOC26{Vgph#IYb`IDHUTxLWq&binlBF5@Q)v_WH)Vs2NcQXH4Bq@->HGm(k_86e3ic9u+v?#P&pXlq^@9%e?r1=R5V_uH3WT-bK&E*i<2@X}0`t8~PoYHiy@yDA+5I=dW661EdM*BnWiJF+5hH08I$>On3i2E~G`1oxfStX)+kqiNXNA`y16PD@bUj#wd`GQkb6-KD7-`N3Kf`395lu5j8n^E(C+HDUDJM8?u8KR+4SfwR~qtG4@-6QWq~+7-k8$IxeC7Pb<Xq%SQZd7SV&E)%Ku8_>w00W^$DVZIgh!v6gk2J0$*4c@t|Gbc`mgrdoL#aW_3R49<Thq!r49SwVXfo&!N?(G5F!ePng%t2y7fQdBlUje+;2k+&Lkj;uM*x{(zUh0z$w;Q+EA69O^7RhA#wXn3|QEo?AVf}1XAbAQ#8*h|7J9yxC(b@OqD>nuDqLH*S!0=S7DN_#A8DDO-26^Tmn!M-zU55l-`j|t`+ZyG3)u&V*;cbg-K^vDPE^5w~c}djENi7jKw=S9jLGik0Y~MpJ&i!Y~yL6b^3)PW{9ktTa_EdR72=4?ps4&I!5xv!y{!X|VgcuSQ=m;@IM`KY4qZ{f2y`Yryw*mi?^A{iRclkznj1zsMihF6b@MjllXRKKEHz3JJB%ARml!4^-zFO&(sNH_3&DRL_I%>|STSnOb2*IpZ3Q&Ck_h71V;6kz7eAdT`Xcj&TgvOL_2iX*^S^1&qsRO_Wdk}7SsZP^#*}yDWuMAM4dZba!JEu0ES3d#3aZrgORDR2St&2boZ-0@PC`TWmn5BcXEtTmPxqDz}d~LYqR%|rxZUy3iqpCoh^XE7JylL<3QYXR*%b?l>(1%;Et;U<9)|t2TS&79?hZdtnrsHP5%LZB3>t{pAeZjg$n}LUPdfpLHCW8b-lo4-%6Ry(ITv<q%InwG-+uBVtszyH+wavGLL?Ch#jH$HAV|4|59uJZ9&V*V>RAwwojZqueqb(X-Q6?tN)x=3rp|P=-hVAjCqP!bpi+V@=la_DRJYR&fz0$zxbBhwiO}INRA3{e--C~Ir$2-T9+d`rqg~BTWL%1yc%H#db$b=dahpMDkA0oM`0XT9M&2GWv+*x)7v}+#@nan@Y_|@!v_XxBd{$5*p6(~|`y=#%=JOi{MzDU+7sa_o|cC)NC8@N9`fYnCR;dDb3re}+!I2bwTw9y7x?^5mEL@XH;Fu}r0p8`7GRh;ng*(a9$0xC2L=iEte>J}O91zTyBe^q;+trK1VJ4iO?C=vWwqPcGJi?<3utOM)=bw@7X7Gwl!YYSi~eFH;CWoZJ2fM(<_zCn}34b^0n?zBAJ*Xd|nJK6f$$$IQ)54bZw+ygl)t~(ZhqXc~uAQ~k-4P;6}z#!}C9!HdUR`hlSyR}4vuPP75Sh7bz_Cl5XhjY8Y_(`faic4LuHhw8iDuEp-d2)860n#3=dJGO8x@y?{IO`E>pimxIpd}d6F5}WJu<QROz8Q;yyd9ezN{YG35}Xeiu3b+8nB|q{K7B0e37JFRJli9p=XAjq1Jj2QW!aP(VT?(lTaGjL*MGCU6yx$3&PS#T{arYM>2NynJG8xRz9xc;248oS3Julz^N2>5VTWk|`gLfi&x&LSqRmhWJA7E)D6q{4l9?ko%$hi}wkh7`;1Ux}YJG~g6f5kqC`L5glcZX7_2o&(5_$sKkI)kxwds+xWJS4WEyv*jK(4H>aCrAd-sTmO&0uZrN?C|a)}_C<wl|b4b{@z`)$BSUX?0A_h%wX|cy7WBn#mI=<R1`{pr^BwnFBm9crYW4<mDL2n<=5DufJvokXa~lZXxY#JU*~D%!%=4Miirn%gfkBPo0Vl-yarIQL{rU41Wo1mYFJCA9`Vr&=dLYWccoQ0atN~N5~0AF(f-o-Cs(00eP@Qv}1iT#hT>Qoy9*AOQ?~wc{kHY+$@IBIf99w#1d%$?8}!}VjT_u>t$Hwh^g6ERahdLbB0@C55-6edrqke^Ua{Kb*D}RxkPeB7|-y|uh%V{kaHAwL*~B1Xs0}(XL7!EhXb`{AYDt*X+(&o#vOQwa&j5zL_1$MJ-)NrBJoIjLl<80h}GRoel(_$ucV53%p6f}@X4mcBMB3lUq)7u@%m+C;rW$k0Jz3=lcHEp@iblRP?nlL7RcNMuA?81yeRltO1p<J+UGNHEg6I}(pQ?Ua|zmGoRD5tc2=4VN83K=Vzci$I&rlCCXIU*?er44$4|v=`|~=%^h=hYD<<%O{Fj0SH$;w%83uO^`+0}WeN!c1p%jtj2}O3(W9x){kykJ2Fx-@)b|}vCtO1rtC4>BW_01R@<eOhFSAS<Xw<kJksfVVQ!wRKqdG0y!6FnEF38*&6dg?*Sejv-o>;7aK4uZh()5Dy9N%Y0*fEpNsywgnsOw*n+Z!(@eost;FVLNwl5bF*WQ>_SlPmc)VNrbYPV-!K#5lzes2C``xP(kn6ucfJK4zQDDe+36Bd49VrY`;Y{{b`Yf&X8TuYP0G)Pcmn=p>1*z^KA`D#j;HyFmi+kp`|Rc;f1A;OavhK^xlSeaCfAfj>;Is*Wi_UlrY%>G{t{Cl&*5Q3PPb(ttC37YpkHuf&K)#_8aI5`=I_zDEg&869o3<P9zQP>C`YA5ql1yXFeq>k-)v)5LbF7(jPJ4<eT>)?dA?`VkleOu=tVyBh_nzVBbaKE+hn6IY+vib#*f5_h^ZA7u_P(hpTWs>JDKKKq$lk*9_@jzt~QZugX&q!^j7I<sNvKx<`az#K3Ooyt9w>x+biP&k9FCOT;R`=Ea0L-6C{9o=?tbb-xfA)KmL69{xlMUko{&O}7h4lAsGH_p1Cm^iHM4khBBY10@UV&BFVviWp)G&?Nx6+{Xp065s`0fZQPPZYf4Fp+!zt16iI1g^oaA(Mgl!cb{r?=K{2#h)8k||4Df(Vb~(ceOu^4R!CUu!-}CPBl;0d%>nQ!&K4dFBCFeOY7$c5!rV9PF06z>R`-t0oeRFm>T<!emt5llu%huwqGVB-Z}bAnCAER$wt<LV*ul_?Dib5je*M!$_n#iaup?_&=EC$zC8f4_h<FKy%lU>&N{WI<D|0Fc#?#9<TThv=Bf&HVOib=D4a!uc_XU3-w#rZ96fp+q#y7zqd*oh`%4luJtXnf}NS+lY+kM`|;ANLS_!leg{@{#PP>Ol#yU_uOKCbN)W@IaH&>=vQkN}PP)=WG5WN9lVwlL<$$r_AYKvOc9ENcwed-Z{mO?O?zW0xC`P5Hq(-@r@{N_r5bi@)Vx+#v<s2eC{#w{DbgSOreJm7(c5^P#*W<g$VrCZl|<=Z@&bajP^hT$%Tk*F#(b!AnUW5Icz|znE{_=uSEIzE_V7Ph%1dPv!8-P*3KjjxGFb=A)$~UTkNQOcTCFmFe-baca{FjW(^X?988vnIk-PW9GuJx%rr?gn2%Lp3Tdcd~oP|g(6}gnTI025?E|0l)pnM1RnCj^XdGe_J3G*!dF?HkU^*4u~Xq|WXIU!t-=CxeMqu$o-j`&+eVxrpuc6X2JH6P_qv5VIia>rnXvA+90PX{+(085@cK#nGT%mG&9aZ(3eQ>XZF4AwGlGC7;9hR>d1L;9q=UcdFX1=Q+fSM=P8~UH%yp4X@ZC5(sn<-xX<XjTIA&Q`du3SQt%rORvau#c2<k1F!-3K&!X`3jMdC_`lD@k^GGx=wf2WH`Z3W7@(nU<DAf3O+5`>nwL(3zkPlWkl!Ok0O+X2gWsKknAD0n@snda(|n<cb47vZ1xkgxmEFtg@EXw$wCdK|ODmM2r5%!!FVVmC@(0p6=6PVRpm{adB|z1=KSd3unkWKG777Ej4bMCqBaqP(FRDD~8gFyO6(F%4N;=s&`WZu)Qw{fDuLC!t<54Jfay=~stPS0ocnJw#I$IThNu*q!MPUOt<mc#+=x*$Q20C?Hm_H%J1&Cu9d8QH~P;n7-U%0;{R3-Cqj(jv#^(J`!Sifv_kn$NVEjD2BznL|(A}8=`z+Q4^7Xk%L?fjj>lL6^7?1=!8sC>2Cl3P-I&qm$~uimUEhqq{XLo(~g%gcD?`PAe}Yj@1|_?p_uKQC+M4FZPaB{=i5n1GpUT=7Fk7im?`e(r1XahqxQ#SrV55<1nPBQ+HwtL$OaWn31aLN)JNrcgKVd9KMUZ8c_;Zb7Cs||Py%-*1-X%|Z2w^+xHTuVfJw~;tvK#XR(B3HwKq^1n|x=8uO`9DE*Sro2spC8Z2f47mpq2A-Jtcy3Os@U;r7JJC>=MYYhD$sbQmj2<o{od6y_#b6=U_d?HaWWm$V~LVUy3mzNi?bTj=RNeoP2to`KUv^jdhqulfq!^Rq%zgk3d-Cam^&_sz%hjKF-|&z<Z{<r*bQqe=aMzIlottX|fxF3$T6C)raBUMsp1`TD0=gu7uUvk0xHyfF!z$lDN1-mI)9x);u=%pUaQy?%ZIMKQAqhc_Bx!OEH6x7RTF^-fz(%Nu0Q5#qve+1R3_gN?l&(>uZ=xK|>T)>Hf3$*ej^m|x(vys0)-@4JU2r#+G!gUO}@`Zl3bM3d57>6?*m;HBUGGAL9$`(P3fRi9r@az2k>H;EF4IvnU-*S&WYmiYgbbtOp-!Z7edbMOM#JT8BqQZKSnxg}QvmelGcMzV}#rXN--qRhwfKNf4k_gJBbpzp8LsLa>r)`Yu~rq=*1j^coGWhOu`zK+9ELa_9FhOu>0vKtwDt|Bp4rF%rgN<HI9oF0dRQzGYp#F&q#N&(%3Y&ryk4{12rLERp|0&O()81@t0&GQ2%{Bax$m4QO`sS{c;4I$X8M%Tqo^L1julD#-l^P{3QiFO_>D8_$_m}dc2M$whM*x}rH5r_r8m^maTiit0SCC4$0!{Icxmxawy8YjT>&`CpE2CH}4#i-SC0&L4Rz_9ssRB4$ezfwc)m{*W2r*YHEoZGmfnh6UwWPdQV&cctsY<O*BCz#BOb{r4?)^;ljO4BreJRS)&u^X6wc`s{O|DDm%N%kU!4L{=jgOJ9d-kHm>Q^~J(8q@wPvP*-8?-@stsw47|nuqNx-A@&Kb#Imi?L}G@RCXJ$w)7g=Ma4_rXiq$N)CO*yC#U>U%qnh$vFP5OJR+aBOD>6##13yzpLj{P<v;xUN8Le*ixN*F*1P6C5l54rIEZmU5OYjk)EUY8J6eLkER=VM_k($dUuZ)9n3s8;=LKWIarRgLc3@0K%!KaqF+#P3R>i?M)J|r|M5~xS!`?#+L-ro{a#a}VY$CMEjYvzpMb<>27@XR_(ZtjR2ROra!RgeutPDz(`W*D&eWcnDPQAshEmy90dhj&pUU+dT4rH7y;!E-S=N~WK34Q'
_CG_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _cg_auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    if len(p) == 0 or len(n) == 0: return float('nan')
    r = np.concatenate([p, n]).argsort(kind='stable').argsort().astype(float) + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def _cg_bits(model, sl, seqs, bs=64):
    """seqs: (bytes, a, b) -> total bits the trunk spends on bytes [a, b) (trimmed from the LEFT to fit the window)"""
    from platypus.inference.closed_form import _ws_batch, _ws_run
    from platypus.inference.jepa import cfg_seq_
    L0 = int(cfg_seq_(model)) - 1; out = np.zeros(len(seqs))
    for i in range(0, len(seqs), bs):
        ch = []
        for s_, a, b in seqs[i:i + bs]:
            cut = max(0, len(s_) - L0); ch.append((s_[cut:], max(1, a - cut), b - cut))
        X, _ = _ws_batch(model, [c[0] + b' ' for c in ch])
        with torch.no_grad(): lp = F.log_softmax(_ws_run(model, X, sl).float(), -1)
        for j, (c, a, b) in enumerate(ch):
            t = torch.arange(a, b, device=lp.device); out[i + j] = float(-lp[j, t - 1, X[j, t]].sum()) / np.log(2)
    return out


def _c3(s):
    t = ' ' + re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip() + ' '; return {t[i:i + 3] for i in range(len(t) - 2)}


def compress_gain(model, cfg, S_slot, vis, S_=None):
    try:
        from platypus.inference.probes import _bench
        G = _bench(cfg); MT = json.loads(zlib.decompress(base64.b85decode(_BENCH_MENT)).decode('utf-8'))
        seen = sorted(set(int(k) for k in (model.__dict__.get('_t2s') or {}).keys())); TH = b'Thing'; rs = np.random.RandomState(int(cfg.seed) + 5151 + vis)
        nmax = int(getattr(cfg, 'inf_compress_n', 600)); JA, JB = [], []; PA, PB, PL = [], [], []; JW = []
        cap = lambda w: w[:1].isupper() or w[:1].isdigit()
        for d in seen:
            sl = int(S_slot(d)); docs = G['docs'].get(str(d), []); ments = MT.get(str(d), [])
            for di in rs.permutation(len(docs)):
                t = docs[int(di)]; ms = ments[int(di)] if int(di) < len(ments) else []; tb = t.encode('latin-1', 'replace')
                sents = [(m.start(), m.end()) for m in re.finditer(r'[^.!?]+[.!?]*', t)]
                sid = lambda x: next((k for k, (a, b) in enumerate(sents) if a <= x < b), 0)
                toks = [(m.start(), m.end(), m.group()) for m in _CG_TOK.finditer(t)]
                inm = {}
                for k, (a, b, w) in enumerate(toks):
                    for mi, (ma, mb, mv) in enumerate(ms):
                        if ma <= a and b <= mb: inm[k] = mi; break
                starts = {k for k, (a, b, w) in enumerate(toks) if any(a == ma for ma, mb, mv in ms)}
                for k in range(1, len(toks)):                                 # (a) joins of two capitalised words, same sentence, one space apart
                    a0, b0, w0 = toks[k - 1]; a1, b1, w1 = toks[k]
                    if not (cap(w0) and cap(w1)) or t[b0:a1] != ' ' or sid(a0) != sid(a1) or len(JA) >= nmax: continue
                    c0 = max(sents[sid(a0)][0], a0 - 60); pre = tb[c0:a0]
                    same = k in inm and (k - 1) in inm and inm[k] == inm[k - 1] and k not in starts
                    JA.append((sl, pre + tb[a0:b1], len(pre) + (b0 - a0) + 1, len(pre) + (b1 - a0))); JW.append((w0, w1, same))
                    JB.append((sl, pre + TH + b' ' + tb[a1:b1], len(pre) + len(TH) + 1, len(pre) + len(TH) + 1 + (b1 - a1), same))
                for j in range(len(ms)):                                      # (b) pairs: each mention with its 8 nearest earlier mentions
                    if len(PL) >= 10 * nmax: break
                    ja, jb, jv = ms[j]; cands = [i for i in range(j) if ms[i][1] <= ja][-8:]
                    for i in cands:
                        ia, ib, iv = ms[i]; si, sj = sents[sid(ia)], sents[sid(ja)]
                        s_i, e_i = max(si[0], ia - 24), min(si[1], ib + 24); s_j = max(sj[0], ja - 48); tg = tb[ja:jb][:40]
                        if sid(ia) == sid(ja) or e_i > s_j:
                            s0 = min(s_i, s_j); w_ = tb[s0:ja]; o_ = tb[s0:ia] + TH + tb[ib:ja]
                        else:
                            w_ = tb[s_i:e_i] + b' ' + tb[s_j:ja]; o_ = tb[s_i:ia] + TH + tb[ib:e_i] + b' ' + tb[s_j:ja]
                        ni, nj = t[ia:ib], t[ja:jb]; wi, wj = set(_CG_TOK.findall(ni.lower())), set(_CG_TOK.findall(nj.lower())); ci, cj = _c3(ni), _c3(nj)
                        PA.append((sl, w_ + tg, len(w_), len(w_) + len(tg))); PB.append((sl, o_ + tg, len(o_), len(o_) + len(tg)))
                        PL.append((iv == jv, len(wi & wj) == 0, len(ci & cj) / max(1, len(ci | cj)), (d, int(di), j)))
        def run(A):
            out = np.zeros(len(A)); bys = {}
            for n, x in enumerate(A): bys.setdefault(x[0], []).append(n)
            for sl, ix in bys.items():
                v = _cg_bits(model, sl, [(A[n][1], A[n][2], A[n][3]) for n in ix])
                for n, b in zip(ix, v): out[n] = b
            return out
        msg = []
        if JW:                                                                # 13 Nov (c) UNITS FROM COUNTS: pointwise mutual information of the two words,
            import collections                                                 #   counted over what the system holds - the store's sentences and the
            uni, big = collections.Counter(), collections.Counter()            #   benchmark documents of the domains read (offline: PMI 0.73 vs the trunk 0.42-0.59)
            txts = [t for d in seen for t in G['docs'].get(str(d), [])]
            for it in ((S_ or {}).get('it') or []):
                try: txts.append(it['s'].decode('latin-1') if isinstance(it['s'], (bytes, bytearray)) else str(it['s']))
                except Exception: pass
            for t in txts:
                ws = _CG_TOK.findall(t)
                for i, w in enumerate(ws):
                    uni[w] += 1
                    if i: big[(ws[i - 1], w)] += 1
            N = max(1, sum(uni.values())); pm = np.array([np.log2((big[(a, b)] + 0.1) * N / (uni[a] * uni[b] + 1e-9)) for a, b, sm in JW]); yy = np.array([sm for a, b, sm in JW])
            msg.append(f"(c) UNITS FROM COUNTS over {len(txts)} texts the system holds ({N} words): PMI AUC same mention vs boundary {_cg_auc(pm[yy], pm[~yy]):.3f}")
        if JA:
            bw = run(JA); bo = run([x[:4] for x in JB]); g = bo - bw; y = np.array([x[4] for x in JB])
            msg.append(f"(a) UNITS, {len(y)} joins of capitalised words ({int(y.sum())} inside one mention): AUC same mention vs boundary - BINDING GAIN {_cg_auc(g[y], g[~y]):.3f} "
                       f"(mean {g[y].mean() if y.any() else float('nan'):.2f} vs {g[~y].mean() if (~y).any() else float('nan'):.2f} bits), plain surprise of the second word {_cg_auc(-bw[y], -bw[~y]):.3f}")
        if PL:
            g = run(PB) - run(PA); y = np.array([p[0] for p in PL]); hd = np.array([p[1] for p in PL]); c3 = np.array([p[2] for p in PL]); vh = hd & (c3 < 0.25)
            ra = (np.argsort(np.argsort(g)) + np.argsort(np.argsort(c3))) / 2.0
            def ante(sc):
                best = {}
                for k, p in enumerate(PL):
                    if p[3] not in best or sc[k] > best[p[3]][0]: best[p[3]] = (sc[k], bool(y[k]))
                has = {p[3] for k, p in enumerate(PL) if y[k]}; return np.mean([best[k][1] for k in has]) if has else float('nan')
            msg.append(f"(b) LINKS, {len(y)} mention pairs ({int(y.sum())} same entity; {int((y & hd).sum())} sharing no word, {int((y & vh).sum())} very hard): AUC same entity vs not - "
                       f"COMPRESSION GAIN all {_cg_auc(g[y], g[~y]):.3f}, HARD {_cg_auc(g[y & hd], g[~y & hd]):.3f}, VERY HARD {_cg_auc(g[y & vh], g[~y & vh]):.3f} "
                       f"(mean {g[y].mean() if y.any() else float('nan'):.2f} vs {g[~y].mean():.2f} bits) | letter triples all {_cg_auc(c3[y], c3[~y]):.3f}, HARD {_cg_auc(c3[y & hd], c3[~y & hd]):.3f} | "
                       f"rank average all {_cg_auc(ra[y], ra[~y]):.3f}, HARD {_cg_auc(ra[y & hd], ra[~y & hd]):.3f} | antecedent right: gain {ante(g):.2f}, letter triples {ante(c3):.2f}, both {ante(ra):.2f}")
        try:                                                                  # 13 Nov (d) ONE RELATION, MANY WORDINGS - SUBSTITUTION: gold relations with
            nm = G['ent_name']; REC = []                                       #   both names in one sentence (subject first, frame <= 12 words); swapping
            for d in seen:                                                     #   frame j into record i: extra bits on i's object (symmetric); same gold
                sl = int(S_slot(d))                                            #   relation type should cost fewer (offline stand-in: AUC 0.58)
                for t in G['docs'].get(str(d), []):
                    for h, tt, fl, ps in G['rel']:
                        H, T = nm.get(str(h), ''), nm.get(str(tt), '')
                        if len(H) < 3 or len(T) < 3 or H not in t or T not in t: continue
                        for s_ in re.split(r'(?<=[.!?])\s+', t):
                            if H in s_ and T in s_ and s_.index(H) + len(H) < s_.index(T):
                                mid = s_[s_.index(H) + len(H):s_.index(T)]
                                if 0 < len(mid.split()) <= 12:
                                    left = s_[max(0, s_.index(H) - 30):s_.index(H)]
                                    REC.append((sl, left.encode('latin-1', 'replace'), H.encode('latin-1', 'replace'), mid.encode('latin-1', 'replace'),
                                                T.encode('latin-1', 'replace')[:30], set(ps), d)); break
            REC = list({(r[1], r[2], r[3], r[4]): r for r in REC}.values())
            if len(REC) >= 20:
                Q = [int(i) for i in rs.permutation(len(REC))[:150]]; PR = []
                for qi in Q:
                    for cj in [int(c) for c in rs.permutation(len(REC))[:16] if int(c) != qi and REC[int(c)][3] != REC[qi][3]][:15]: PR.append((qi, cj))
                base = {}
                for i in {x for p in PR for x in p}: base[i] = None
                bl = list(base); bseq = [(REC[i][0], REC[i][1] + REC[i][2] + REC[i][3] + REC[i][4], len(REC[i][1] + REC[i][2] + REC[i][3]), len(REC[i][1] + REC[i][2] + REC[i][3] + REC[i][4])) for i in bl]
                bb = dict(zip(bl, run(bseq)))
                sw = []
                for i, j in PR:
                    for x, y in ((i, j), (j, i)):
                        pre = REC[x][1] + REC[x][2] + REC[y][3]; sw.append((REC[x][0], pre + REC[x][4], len(pre), len(pre) + len(REC[x][4])))
                v = run(sw); sc = np.array([-((v[2 * n] - bb[i]) + (v[2 * n + 1] - bb[j])) / 2 for n, (i, j) in enumerate(PR)])
                yy = np.array([bool(REC[i][5] & REC[j][5]) for i, j in PR]); top = []; chn = []
                for qi in Q:
                    ix = [n for n, (i, j) in enumerate(PR) if i == qi]
                    if ix: top.append(yy[max(ix, key=lambda n: sc[n])]); chn.append(np.mean(yy[ix]))
                msg.append(f"(d) RELATIONS by substitution, {len(REC)} one-sentence gold-relation records, {len(Q)} queries x up to 15 candidates: AUC same gold type vs not "
                           f"{_cg_auc(sc[yy], sc[~yy]):.3f} | best candidate of the same type {np.mean(top) if top else float('nan'):.3f} (chance {np.mean(chn) if chn else float('nan'):.3f})")
        except Exception as e8:
            msg.append(f"(d) RELATIONS unavailable ({type(e8).__name__}: {e8})"[:160])
        print(f"  [COMPRESSION GAIN | visit {vis + 1}] from the trunk (store off), Re-DocRED documents of the {len(seen)} domains visited, gold mentions || "
              + (" || ".join(msg) if msg else "no mentions yet"), flush=True)
    except Exception as e:
        _fail('inference.compress'); print(f"  [COMPRESSION GAIN] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


def _sl_initials(y):
    return b''.join(w[:1] for w in y.split() if w[:1].isupper())


def store_link(model, cfg, S_, S_slot, vis):
    """13 Nov STORE LINKING (inf_store_link; ONE MODEL CHANGE - it changes the store's real-text evidence record S_['relc'], which feeds the
    reader-vs-gold and real-text relational-module readouts; the trunk's training is untouched). Each sleep, over the sentences stored since
    the last one, a mention (a filler of the fast channel's records) is linked to an earlier, different name among the fillers of the 5
    sentences stored before it (same domain):
      LETTERS      it is the earlier name's last word(s) ('Thatcher' -> 'Margaret Thatcher') or its initials ('AGF' -> 'Asian Games Federation')
      COMPRESSION  where letters cannot decide (a capitalised mention sharing no word with the earlier name): the trunk saves at least
                   inf_store_link_bits bits on the mention when the earlier name's sentence precedes it, against the same sentence with the
                   name replaced by 'Thing' (GPU readout: same entity 3.9 bits, others 0.3); the best such earlier name
    GUIDANCE ONLY ADDS: each record of the sentence is ALSO recorded with its linked names (the originals stay). Readout: links of each kind,
    their precision where both names belong to benchmark entities, records added."""
    try:
        from platypus.inference.units import _chk_records
        RC = S_.get('relc')
        if RC is None: return
        it = S_['it']; st = int(model.__dict__.get('_link_from', 0)); end = len(it); model.__dict__['_link_from'] = end
        bits_min = float(getattr(cfg, 'inf_store_link_bits', 3.0)); nmax = int(getattr(cfg, 'inf_store_link_n', 4000))
        TH = b'Thing'; win = []; jobs = []; links = {}; nlet = 0; seen_s = 0
        for i in range(st, end):
            x_ = it[i]
            if x_.get('gone') or not isinstance(x_.get('s'), (bytes, bytearray)): continue
            s = bytes(x_['s']); d = int(x_.get('d', 0)); recs = _chk_records(cfg, S_, s); seen_s += 1
            win = [w for w in win if w[0] == d and i - w[1] <= 5]
            fil = {f for a, r, b in recs for f in (a, b)}
            for xf in fil:
                if not xf[:1].isupper(): continue
                xs = xf.split(); hit = None
                for wd, wi, ws, wf in reversed(win):
                    for y in wf:
                        if y == xf or len(y) <= len(xf): continue
                        ys = y.split()
                        if ys[-len(xs):] == xs or (len(xs) == 1 and len(xf) >= 2 and xf.isupper() and _sl_initials(y) == xf): hit = y; break
                    if hit: break
                if hit is not None: links[(i, xf)] = (hit, 'L'); nlet += 1; continue
                if len(jobs) >= 2 * nmax or len(xs) > 3: continue
                px = s.find(xf)
                if px < 0: continue
                cands = []
                for wd, wi, ws, wf in reversed(win):
                    for y in wf:
                        if y != xf and y[:1].isupper() and not (set(y.split()) & set(xs)) and len(cands) < 4: cands.append((y, ws))
                for y, ws in cands:
                    py = ws.find(y)
                    if py < 0: continue
                    snip_w = ws[max(0, py - 24):py + len(y) + 24]; snip_o = ws[max(0, py - 24):py] + TH + ws[py + len(y):py + len(y) + 24]
                    pre = s[max(0, px - 48):px]; sl = int(S_slot(d))
                    jobs.append((i, xf, y, sl, snip_w + b' ' + pre + xf, len(snip_w) + 1 + len(pre), snip_o + b' ' + pre + xf, len(snip_o) + 1 + len(pre)))
            win.append((d, i, s, fil))
        ncmp = 0; gains = []
        if jobs:
            bysl = {}
            for n, j in enumerate(jobs): bysl.setdefault(j[3], []).append(n)
            gw = np.zeros(len(jobs)); go = np.zeros(len(jobs))
            for sl, ix in bysl.items():
                gw[ix] = _cg_bits(model, sl, [(jobs[n][4], jobs[n][5], len(jobs[n][4])) for n in ix])
                go[ix] = _cg_bits(model, sl, [(jobs[n][6], jobs[n][7], len(jobs[n][6])) for n in ix])
            best = {}
            for n, j in enumerate(jobs):
                g = go[n] - gw[n]
                if g >= bits_min and (j[0], j[1]) not in links and g > best.get((j[0], j[1]), (-1e9, None))[0]: best[(j[0], j[1])] = (g, j[2])
            for k, (g, y) in best.items(): links[k] = (y, 'C'); gains.append(g); ncmp += 1
        nadd = 0; byi = {}
        for (i, xf), (y, kd) in links.items(): byi.setdefault(i, {})[xf] = y
        for i, mp in byi.items():
            for a, r, b in _chk_records(cfg, S_, bytes(it[i]['s'])):
                a2, b2 = mp.get(a, a), mp.get(b, b)
                if (a2, b2) != (a, b) and a2 != b2: RC.setdefault((a2, r), {}).setdefault(b2, []).append(i); nadd += 1
        prec = {'L': [], 'C': []}
        if str(getattr(cfg, 'inf_bench', '') or ''):
            from platypus.inference.probes import _bench
            t2e = _bench(cfg)['_t2e']
            def ents(x):
                es = set()
                for w in x.decode('latin-1').split(): es |= t2e.get(w, set())
                return es
            for (i, xf), (y, kd) in links.items():
                ex, ey = ents(xf), ents(y)
                if ex and ey: prec[kd].append(bool(ex & ey))
        tot = model.__dict__.setdefault('_link_tot', [0, 0, 0]); tot[0] += nlet; tot[1] += ncmp; tot[2] += nadd
        print(f"  [STORE LINKING | visit {vis + 1}] {seen_s} sentences stored since the last sleep | LETTER links {nlet} (on benchmark entities {len(prec['L'])}, "
              f"same entity {np.mean(prec['L']) if prec['L'] else float('nan'):.2f}) | COMPRESSION links {ncmp} of {len(jobs)} candidate pairs (gain >= {bits_min:g} bits; "
              f"median {np.median(gains) if gains else float('nan'):.1f}; on benchmark entities {len(prec['C'])}, same entity {np.mean(prec['C']) if prec['C'] else float('nan'):.2f}) | "
              f"records ADDED with linked names {nadd} | so far: letter {tot[0]}, compression {tot[1]}, records added {tot[2]}", flush=True)
    except Exception as e:
        _fail('inference.compress:store_link'); print(f"  [STORE LINKING] unavailable ({type(e).__name__}: {e})"[:240], flush=True)


_PQ_NAME = re.compile(rb"(?<![.!?] )(?<!^)\b[A-Z][A-Za-z'\-]{2,}\b")


def preq_bits(model, cfg, vals, seen_d, next_d, slot_of, S_, tag, nb=8):
    """13 Nov READOUT: PREQUENTIAL BITS ON KNOWLEDGE (inf_preq) - does what sleep produced (replay, links, written conclusions) make the
    model predict text it has NEVER trained on, replayed or dreamed? Each domain's held-out validation text (fixed batches), from the
    weights alone (store and name memory off): bits per byte on ALL bytes, on the bytes of capitalised names the store ALREADY KNOWS (in its
    name index) and on names it does not know - knowledge shows as lower bits on known names. For the domains read so far (held out) and
    for the NEXT domain (in pass 1: text not yet read at all - prediction before learning). Compare runs on the same seed."""
    try:
        from platypus.inference.store import _INFM
        from platypus.capacity import apply_delta, domain_delta
        idx = set((S_ or {}).get('idx', {}).keys()); Ts = int(model.cfg.seq_len); parts = []
        doms = [(d, 'read') for d in seen_d] + ([(next_d, 'NEXT, not yet read')] if next_d is not None and next_d not in seen_d else [])
        had = 'force' in _INFM; sv = _INFM.get('force'); _INFM['force'] = False
        try:
            for d, kind in doms:
                g = vals[d]; cur = model.__dict__.get('_cur_dom')                # the NEXT domain has no trained slot yet: read it through the slot
                sl = int(slot_of(d)) if kind == 'read' or cur is None else int(slot_of(int(cur)))   #   of the domain just visited (13 Nov fix)
                if hasattr(g, 'reset'): g.reset()
                dd = domain_delta(model, sl) if (getattr(model, '_spec', None) and sl != getattr(model, '_live_dom', None)) else {}
                apply_delta(model, dd, +1); model.eval(); acc = {'all': [0.0, 0], 'known': [0.0, 0], 'unknown': [0.0, 0]}
                try:
                    with torch.no_grad():
                        for _ in range(nb):
                            x, y = g()
                            lg = model(x, sl)[:, -Ts:]; yy = y[:, -Ts:]
                            nl = (F.cross_entropy(lg.reshape(-1, 256).float(), yy.reshape(-1), reduction='none') / np.log(2)).view(yy.shape).cpu().numpy()
                            yb = yy.cpu().numpy().astype(np.uint8)
                            for r in range(yb.shape[0]):
                                row = bytes(yb[r]); acc['all'][0] += float(nl[r].sum()); acc['all'][1] += len(row)
                                for m in _PQ_NAME.finditer(row):
                                    k = 'known' if m.group(0) in idx else 'unknown'
                                    acc[k][0] += float(nl[r, m.start():m.end()].sum()); acc[k][1] += m.end() - m.start()
                finally:
                    apply_delta(model, dd, -1); model.train()
                f = lambda k: acc[k][0] / max(1, acc[k][1])
                parts.append((d, kind, f('all'), f('known'), acc['known'][1], f('unknown'), acc['unknown'][1]))
        finally:
            if had: _INFM['force'] = sv
            else: _INFM.pop('force', None)
        rd = [p for p in parts if p[1] == 'read']; nx = [p for p in parts if p[1] != 'read']
        wavg = lambda L, i, j: sum(p[i] * p[j] for p in L) / max(1, sum(p[j] for p in L))
        msg = (f"domains READ ({len(rd)}): all bytes {np.mean([p[2] for p in rd]):.4f} | KNOWN names {wavg(rd, 3, 4):.3f} ({sum(p[4] for p in rd)} bytes) | "
               f"unknown names {wavg(rd, 5, 6):.3f} ({sum(p[6] for p in rd)} bytes)") if rd else "no domain read yet"
        if nx:
            p = nx[0]; msg += f" || NEXT domain {p[0]} (not yet read): all bytes {p[2]:.4f} | KNOWN names {p[3]:.3f} ({p[4]} bytes) | unknown names {p[5]:.3f} ({p[6]} bytes)"
        print(f"  [PREQUENTIAL BITS | {tag}] held-out text never trained on, replayed or dreamed; weights only; bits per byte || {msg} | store knows {len(idx)} names", flush=True)
    except Exception as e:
        _fail('inference.compress:preq'); print(f"  [PREQUENTIAL BITS] unavailable ({type(e).__name__}: {e})"[:240], flush=True)
