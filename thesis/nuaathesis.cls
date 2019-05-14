%%
%% This is file `nuaathesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% nuaathesis.dtx  (with options: `cls')
%% ----------------------------------------------------------------
%% nuaathesis --- Thesis Template for Nanjing University of Aeronautics and Astronautics
%% Licensed under the Apache License, Version 2.0
%% See http://www.apache.org/licenses/LICENSE-2.0
%% ----------------------------------------------------------------
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nuaathesis}
[2018/11/01 v2.1 NUAA Thesis Template]
\InputIfFileExists{nuaathesis.cfg}{}{}
\AtEndOfClass{
\newcommand\nuaa@label@nuaa{南京航空航天大学}
\newcommand\nuaa@label@nuaajc{南京航空航天大学金城学院}
\newcommand\nuaa@label@worktype@paper{毕业论文}
\newcommand\nuaa@label@worktype@design{毕业设计}
\newcommand\nuaa@label@worktype@master{硕士学位论文}
\newcommand\nuaa@label@worktype@doctor{博士学位论文}
\newcommand\nuaa@label@thesisnum{编号}
\newcommand\nuaa@label@title{题\quad 目}
\newcommand\nuaa@label@teamname{团队名称}
\newcommand\nuaa@label@author{学生姓名}
\newcommand\nuaa@label@studentid{学\hfill 号}
\newcommand\nuaa@label@college{学\hfill 院}
\newcommand\nuaa@label@department{系\hfill 部}
\newcommand\nuaa@label@major{专\hfill 业}
\newcommand\nuaa@label@classid{班\hfill 级}
\newcommand\nuaa@label@adviser{指\hfill 导\hfill 教\hfill 师}
\newcommand\nuaa@label@researchername{研究生姓名}
\newcommand\nuaa@label@majorsubject{学科、专业}
\newcommand\nuaa@label@researchfield{研\hfill 究\hfill 方\hfill 向}
\newcommand\nuaa@label@professionaltype{专\hfill 业\hfill 类\hfill 别}
\newcommand\nuaa@label@professionalfield{专\hfill 业\hfill 领\hfill 域}
\newcommand\nuaa@label@graduateschool{研究生院}
\newcommand\nuaa@labelEn@nuaa{Nanjing University of Aeronautics and Astronautics}
\newcommand\nuaa@labelEn@graduateschool{The Graduate School}
\newcommand\nuaa@label@abstract{摘\quad 要}
\newcommand\nuaa@label@abstractshort{摘要}
\newcommand\nuaa@label@keywords{关键词：}
\newcommand\nuaa@label@keywordsep{，}
\newcommand\nuaa@label@abstract@toc{摘要}
\newcommand\nuaa@labelEn@abstract{Abstract}
\newcommand\nuaa@labelEn@ABSTRACT{ABSTRACT}
\newcommand\nuaa@labelEn@KeyWords{Key Words: }
\newcommand\nuaa@labelEn@keywords{Keywords: }
\newcommand\nuaa@labelEn@keywordsep{, }
\newcommand\nuaa@labelJa@abstract{要 旨}
\newcommand\nuaa@labelJa@keywords{キーワード：}
\newcommand\nuaa@labelJa@keywordsep{、}
\newcommand\nuaa@label@reportpaper{毕业设计（论文）报告纸}
\ifnuaa@lang@cn
  \newcommand\listfiguretablename{图表清单}
  \def\equationautorefname{式}
  \def\AMSautorefname{式}
\else\ifnuaa@lang@en
  \newcommand\listfiguretablename{List of Figures and Tables}
\else\ifnuaa@lang@ja
  \newcommand\listfiguretablename{図 表 目 次}
  \renewcommand\bibname{参考文献}
  \def\equationautorefname{式}
  \def\AMSautorefname{式}
\fi\fi\fi
}
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
  family=nuaa,
  prefix=nuaa@,
  setkeys=\kvsetkeys
}
\newif\ifnuaa@bachelor \nuaa@bachelorfalse
\newif\ifnuaa@master   \nuaa@masterfalse
\newif\ifnuaa@doctor   \nuaa@doctorfalse
\define@key{nuaa}{degree}{
  \expandafter\csname nuaa@#1true\endcsname}
\DeclareBoolOption[false]{zhuanshuo}
\newif\ifnuaa@lang@cn \nuaa@lang@cnfalse
\newif\ifnuaa@lang@en \nuaa@lang@enfalse
\newif\ifnuaa@lang@ja \nuaa@lang@jafalse
\define@key{nuaa}{lang}{
  \expandafter\csname nuaa@lang@#1true\endcsname}
\newif\ifnuaa@worktype@paper  \nuaa@worktype@paperfalse
\newif\ifnuaa@worktype@design \nuaa@worktype@designfalse
\define@key{nuaa}{type}{
  \expandafter\csname nuaa@worktype@#1true\endcsname}
\DeclareBoolOption[false]{blankleft}
\DeclareBoolOption[false]{abstractopenright}
\DeclareStringOption{fontset}
\DeclareBoolOption[false]{nobold}
\DeclareDefaultOption{}
\kvsetkeys{nuaa}{}
\ProcessKeyvalOptions*
\ifnuaa@bachelor\relax\else
\ifnuaa@master\relax\else
\ifnuaa@doctor\relax\else
  \ClassError{nuaathesis}{
    Thesis Degree must be specified: \MessageBreak
    degree=[bachelor|master|doctor]}
\fi\fi\fi
\ifnuaa@bachelor
  \ifnuaa@worktype@paper\relax\else
    \ifnuaa@worktype@design\relax\else
      \ClassError{nuaathesis}{
        Thesis Type must be specified: \MessageBreak
        type=[paper|design]}
    \fi
  \fi
\else
  \ifnuaa@worktype@design
    \ClassError{nuaathesis}{You should submit paper instead of design}
  \else
    \nuaa@worktype@papertrue
  \fi
\fi
\ifnuaa@lang@cn\relax\else
  \ifnuaa@lang@en\relax\else
    \ifnuaa@lang@ja\relax\else
      \nuaa@lang@cntrue
    \fi
  \fi
\fi
\iffalse
  \newcommand\nuaa@university{\nuaa@label@nuaajc}
  \newcommand\nuaa@universityLogo{nuaa-jc.jpg}
\else
  \newcommand\nuaa@university{\nuaa@label@nuaa}
  \newcommand\nuaa@universityLogo{nuaa.pdf}
\fi
\newcommand\nuaa@worktypecn{%
  \ifnuaa@bachelor%
    \ifnuaa@worktype@paper%
      \nuaa@label@worktype@paper%
    \else%
      \nuaa@label@worktype@design%
    \fi%
  \else%
    \ifnuaa@master%
      \nuaa@label@worktype@master%
    \else%
      \nuaa@label@worktype@doctor%
    \fi%
  \fi%
}
\def\nuaaset{\kvsetkeys{nuaa@value}}
\def\nuaasetEn{\kvsetkeys{nuaa@valueEn}}
\def\nuaasetJa{\kvsetkeys{nuaa@valueJa}}
\def\nuaa@define #1{
  \define@key{nuaa}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname #1\endcsname##1{
    \expandafter\gdef\csname nuaa@#1\endcsname{##1}}
  \csname #1\endcsname{}
}
\def\nuaa@define@list#1#2{
  \define@key{nuaa}{#1}{\csname #1\endcsname{##1}}
  \expandafter\gdef\csname nuaa@#1\endcsname{}
  \expandafter\gdef\csname nuaa@#1@pdf\endcsname{}
  \expandafter\gdef\csname #1\endcsname##1{
    \@for\reserved@a:=##1\do{
      \expandafter\ifx\csname nuaa@#1\endcsname\@empty\else
        \expandafter\g@addto@macro\csname nuaa@#1\endcsname{%
          \ignorespaces #2}
        \expandafter\g@addto@macro\csname nuaa@#1@pdf\endcsname{,}
      \fi
      \expandafter\expandafter\expandafter\g@addto@macro%
        \expandafter\csname nuaa@#1\expandafter\endcsname\expandafter{\reserved@a}
    }
    \expandafter\gdef\csname nuaa@#1@pdf\endcsname{##1}
  }
}
\nuaa@define{value@title}
\nuaa@define{value@author}
\nuaa@define{value@college}
\nuaa@define{value@applydate}
\ifnuaa@bachelor
  \nuaa@define@list{value@advisers}{、}
\else
  \nuaa@define@list{value@advisers}{\linebreak}
\fi
  \nuaa@define{value@major}
  \nuaa@define{value@studentid}
  \nuaa@define{value@classid}
  \nuaa@define{value@libraryclassid}
  \nuaa@define{value@subjectclassid}
  \nuaa@define{value@thesisid}
  \nuaa@define{value@majorsubject}
  \nuaa@define{value@researchfield}
\nuaa@define{valueEn@title}
\ifnuaa@bachelor\relax
\else
\fi
  \nuaa@define{valueEn@college}
  \nuaa@define{valueEn@majorsubject}
  \nuaa@define{valueEn@author}
  \nuaa@define{valueEn@advisers}
  \nuaa@define{valueEn@degreefull}
  \nuaa@define{valueEn@applydate}
\nuaa@define{valueJa@title}
\RequirePackage{etoolbox}
\RequirePackage{environ}
\newcommand{\nuaa@@abstract}[1]{\long\gdef\nuaa@abstract{#1}}
\newenvironment{abstract}{\Collect@Body\nuaa@@abstract}{}
\newcommand{\nuaa@@abstractEn}[1]{\long\gdef\nuaa@abstractEn{#1}}
\newenvironment{abstractEn}{\Collect@Body\nuaa@@abstractEn}{}
\newcommand{\nuaa@@abstractJa}[1]{\long\gdef\nuaa@abstractJa{#1}}
\newenvironment{abstractJa}{\Collect@Body\nuaa@@abstractJa}{}
\nuaa@define@list{keywords}{\nuaa@label@keywordsep}
\nuaa@define@list{keywordsEn}{\nuaa@labelEn@keywordsep}
\nuaa@define@list{keywordsJa}{\nuaa@labelJa@keywordsep}
\ifnuaa@lang@cn
  \newcommand\nuaa@title{\nuaa@value@title}
\else\ifnuaa@lang@en
  \newcommand\nuaa@title{\nuaa@valueEn@title}
\else\ifnuaa@lang@ja
  \newcommand\nuaa@title{\nuaa@valueJa@title}
\fi\fi\fi
\newcommand\nuaa@font@toc{\normalsize}
\RequirePackage{expl3}
\ExplSyntaxOn
\sys_if_engine_xetex:TF{
  \PassOptionsToPackage{no-math}{fontspec}
}{}
\ExplSyntaxOff
\ifnuaa@lang@cn
  \DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexbook}}
  \ProcessKeyvalOptions*
  \ifnuaa@bachelor
    \PassOptionsToClass{zihao=-4,linespread=1.625}{ctexbook}
  \else
    \PassOptionsToClass{zihao=5,linespread=1.5873}{ctexbook}
  \fi
  \PassOptionsToClass{a4paper,scheme=chinese,space=auto,UTF8}{ctexbook}
  \ifx\nuaa@fontset\@empty\relax
    \PassOptionsToClass{fontset=\nuaa@fontset}{ctexbook}
  \fi
  \LoadClass{ctexbook}
  \ctexset{chapter = {
    lofskip = 0pt,
    lotskip = 0pt
  }}
  \newcommand\nuaa@font@title{\sffamily\heiti}
  \newcommand\nuaa@indentloft{3.5pc}
  \newcommand\nuaa@chaptername\CTEX@chaptername
  \PassOptionsToPackage{indentafter}{titlesec}
\else\ifnuaa@lang@en
  \DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{book}}
  \ProcessKeyvalOptions*
  \PassOptionsToClass{a4paper}{book}
  \LoadClass{book}
  \newcommand\nuaa@font@title{\sffamily}
  \newcommand\nuaa@indentloft{5.0pc}
  \newcommand\nuaa@chaptername{\@chapapp\space \thechapter}
  \PassOptionsToPackage{indentafter}{titlesec}
\else\ifnuaa@lang@ja
  \DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ujbook}}
  \ProcessKeyvalOptions*
  \PassOptionsToClass{uplatex,a4paper}{ujbook}
  \LoadClass{ujbook}
  \RequirePackage[deluxe, uplatex]{otf}
  \newcommand\nuaa@font@title{\sffamily\gtfamily}
  \newcommand\nuaa@indentloft{3.5pc}
  \newcommand\nuaa@chaptername{\@chapapp\thechapter\@chappos}
  \PassOptionsToPackage{indentafter}{titlesec}
\fi\fi
  \ifnuaa@bachelor
    \PassOptionsToPackage{zihao=-4,linespread=1.625}{ctex}
  \else
    \PassOptionsToPackage{zihao=5,linespread=1.5873}{ctex}
  \fi
  \PassOptionsToPackage{scheme=plain}{ctex}
  \ifx\nuaa@fontset\@empty\relax
    \PassOptionsToPackage{fontset=\nuaa@fontset}{ctex}
  \fi
  \RequirePackage{ctex}
  \typeout{Patching list of figure/table in chapter}
  \patchcmd{\@chapter}
    {\addtocontents{lof}{\protect\addvspace{10\p@}}}
    {}
    {\typeout{lof-ok}}
    {\typeout{lof-FAIL}}
  \patchcmd{\@chapter}
    {\addtocontents{lot}{\protect\addvspace{10\p@}}}
    {}
    {\typeout{lot-ok}}
    {\typeout{lot-FAIL}}
\fi
\ExplSyntaxOn
\sys_if_engine_xetex:TF{}{
  \sys_if_output_dvi:TF{
    \PassOptionsToPackage{dvipdfmx}{graphicx}
    \PassOptionsToPackage{dvipdfmx}{hyperref}
    \def\pgfsysdriver{pgfsys-dvipdfm.def}
  }{}
}
\ExplSyntaxOff
\ifnuaa@nobold
  \newcommand\nuaa@textbf[1]{#1}
\else
  \newcommand\nuaa@textbf[1]{\textbf{#1}}
\fi
\RequirePackage{geometry}  % 页边距
\RequirePackage{fancyhdr}  % 页眉页脚
\RequirePackage{titlesec}  % 各级标题
\RequirePackage{titletoc}  % 目录
\RequirePackage[numbers,square,comma,super,sort&compress]{natbib}
\PassOptionsToPackage{normalem}{ulem}
\RequirePackage{ulem}      % 英文下划线
\RequirePackage{graphicx}  % 图
\RequirePackage{array}
\RequirePackage{tabu}
\RequirePackage{booktabs}  % toprule, etc
\RequirePackage{multicol}
\RequirePackage{caption}   % DeclareCaptionFont
\RequirePackage{hyperref}
\RequirePackage{ifxetex}
\RequirePackage{siunitx}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{aliascnt} % newaliascnt/aliascntresetthe
\RequirePackage[inline]{enumitem}
\RequirePackage{floatrow} % 替代原 float 包
\floatsetup[table]{style=plaintop}
\ifnuaa@lang@ja
  \RequirePackage{pxjahyper}
\fi
\RequirePackage[defaultsups]{newtxtext}
\RequirePackage{newtxmath}
\ExplSyntaxOn
\cs_set:Npn \zhcn #1{
  \str_if_eq_x:nnTF{\f@family}{\rmdefault}
  {{\songti#1}}{
  \str_if_eq_x:nnTF{\f@family}{\sfdefault}
  {{\heiti#1}}{
  \str_if_eq_x:nnTF{\f@family}{\ttdefault}
  {{\kaiti#1}}{#1}
  }}}
\cs_set:Npn \jpn #1{
  \str_if_eq_x:nnTF{\f@family}{\rmdefault}
  {{\mcfamily#1}}{
  \str_if_eq_x:nnTF{\f@family}{\sfdefault}
  {{\gtfamily#1}}{#1}
  }}
\ExplSyntaxOff
\newif\if@frontmatter
\newif\if@backmatter
\let\nuaa@frontmatter\frontmatter
\let\nuaa@mainmatter\mainmatter
\let\nuaa@appendix\appendix
\let\nuaa@backmatter\backmatter
\renewcommand{\frontmatter}{
  \nuaa@frontmatter
  \@frontmattertrue
  \@backmatterfalse
  \ifnuaa@bachelor\pagenumbering{roman}\else\pagenumbering{Roman}\fi
  \pagestyle{style@main}
}
\renewcommand{\mainmatter}{
  \nuaa@mainmatter
  \@frontmatterfalse
  \@backmatterfalse
  \pagenumbering{arabic}
  \pagestyle{style@main}
  \setlength\leftskip{\nuaaparleft}
}
\renewcommand{\backmatter}{
  \nuaa@backmatter
  \@frontmatterfalse
  \@backmattertrue
  \setlength\nuaaparleft{0pt}
  \setlength\leftskip{\nuaaparleft}
}
\newlength\nuaaparleft
\ifnuaa@bachelor
  \setlength\nuaaparleft{2\ccwd}
\else
  \setlength\nuaaparleft{0pt}
\fi
\newcommand\nuaafontparleft{
  \addtolength\@totalleftmargin{\nuaaparleft}
  \addtolength\linewidth{-\nuaaparleft}
  \parshape 1 \nuaaparleft \linewidth
}
\ifnuaa@bachelor \geometry{
  top=2.5cm,
  bottom=2cm,
  left=2cm,
  right=2cm,
  headheight=0.75cm,
  headsep=2bp,
  %footskip=0.8cm,
  includehead,
  includefoot
} \else \geometry{
  top=3.3cm,
  bottom=3.3cm,
  left=3.0cm,
  right=2.8cm,
  headheight=15.6bp,
  headsep=0.15cm,
  footskip=15.6bp
}
\fi
\newcommand\nuaa@footerpagenum@decorate[1]{%
\ifnuaa@bachelor%
  \if@frontmatter{#1}\else{- #1 -}\fi%
\else%
  {#1}%
\fi%
}
\fancypagestyle{style@empty}{
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{style@main}{
  \fancyhead{}
  \ifnuaa@bachelor
    \fancyhead[L]{
      \setlength{\unitlength}{1mm}
      \begin{picture}(0,0)
        \put(7.3,1.5){\includegraphics[width=6cm]{\nuaa@universityLogo}}
      \end{picture}
    }
    \fancyhead[R]{\songti\zihao{4}\nuaa@label@reportpaper\hspace{1\ccwd}}
  \else
    \fancyhead[C]{\ifodd\value{page}
      {\mbox{\songti\zihao{5}\nuaa@university\nuaa@worktypecn}}
    \else
      {\mbox{\zihao{5}\nuaa@title}}
    \fi}
  \fi
  \fancyfoot{}
  \if@twoside
    \fancyfoot[OR,EL]{\footnotesize{\nuaa@footerpagenum@decorate{\thepage}}}
  \else
    \fancyfoot[R]{\footnotesize{\nuaa@footerpagenum@decorate{\thepage}}}
  \fi
  \renewcommand{\headrulewidth}{0.75bp}
  \ifnuaa@bachelor
    \renewcommand{\footrulewidth}{0.75bp}
  \fi
}
\ifnuaa@bachelor
  \titleformat{\chapter}
    {\centering\linespread{2.41}\nuaa@font@title\fontsize{15bp}{15.6bp}\selectfont}
    {\nuaa@chaptername}{1em}{}
  \titlespacing*{\chapter}{0pt}{23.7bp}{16.5bp}
  \assignpagestyle{\chapter}{style@main}
  \titleformat*{\section}
    {\linespread{1.5}\nuaa@font@title\fontsize{14bp}{20.8bp}\selectfont}
  \titlespacing*{\section}{0pt}{0bp}{0bp}
  \titleformat*{\subsection}
    {\linespread{1.5}\nuaa@font@title\fontsize{12bp}{15.6bp}\selectfont}
  \titlespacing*{\subsection}{0pt}{0bp}{0bp}
  \titleformat*{\subsubsection}
    {\linespread{1.5}\nuaa@font@title\fontsize{12bp}{15.6bp}\selectfont}
  \titlespacing*{\subsubsection}{0pt}{0bp}{0bp}
\else
  \titleformat{\chapter}
    {\centering\linespread{1.0}\nuaa@font@title\fontsize{15bp}{20bp}\selectfont}
    {\nuaa@chaptername}{1em}{}
  \titlespacing*{\chapter}{0pt}{10.4bp}{23.4bp}
  \assignpagestyle{\chapter}{style@main}
  \titleformat*{\section}
    {\linespread{1.0}\nuaa@font@title\fontsize{14pt}{20pt}\selectfont}
  \titlespacing*{\section}{0pt}{7.8bp}{7.8bp}
  \titleformat*{\subsection}
    {\linespread{1.0}\nuaa@font@title\fontsize{12bp}{20bp}\selectfont}
  \titlespacing*{\subsection}{0pt}{7.8bp}{7.8bp}
  \titleformat*{\subsubsection}
    {\linespread{1.0}\nuaa@font@title\fontsize{12bp}{20bp}\selectfont}
  \titlespacing*{\subsubsection}{0pt}{7.8bp}{7.8bp}
\fi
\setcounter{secnumdepth}{3}
\ifnuaa@lang@en
  \titlecontents{chapter}[5pc]
    {\nuaa@font@toc}
    {\contentslabel[\chaptername~\thecontentslabel]{5pc}}
    {\renewcommand\thecontentslabel{\relax}\hspace*{-5pc}}
    {\titlerule*[1ex]{.}\contentspage}
\else
  \titlecontents{chapter}[3.5pc]
    {\nuaa@font@toc}
    {\contentslabel[\thecontentslabel]{3.5pc}}
    {\hspace*{-3.5pc}}
    {\titlerule*[1ex]{.}\contentspage}
\fi
\titlecontents{section}[3pc]
  {\nuaa@font@toc}
  {\contentslabel[\thecontentslabel]{2pc}}
  {}
  {\titlerule*[1ex]{.}\contentspage}
\titlecontents{subsection}[5pc]
  {\nuaa@font@toc}
  {\contentslabel[\thecontentslabel]{3pc}}
  {}
  {\titlerule*[1ex]{.}\contentspage}
\titlecontents{figure}[\nuaa@indentloft]
  {\nuaa@font@toc}
  {\contentslabel[\figurename~\thecontentslabel]{\nuaa@indentloft}}
  {\figurename}
  {\titlerule*[1ex]{.}\contentspage}
\titlecontents{table}[\nuaa@indentloft]
  {\nuaa@font@toc}
  {\contentslabel[\tablename~\thecontentslabel]{\nuaa@indentloft}}
  {\tablename}
  {\titlerule*[1ex]{.}\contentspage}
\pretocmd{\bibliography}
  {\begingroup\linespread{1.0}\fontsize{10.5bp}{15.6bp}\selectfont}
  {}{}
\apptocmd{\bibliography}
  {\endgroup}
  {}{}
\newcommand\nuaa@font@bib{\linespread{1.0}\fontsize{10.5bp}{15.6bp}\selectfont}
\renewcommand\@biblabel[1]{\nuaa@font@bib[#1]}
\renewenvironment{thebibliography}[1]{
  \chapter*{\bibname}
  \list{\@biblabel{\@arabic\c@enumiv}}
  {
    \settowidth\labelwidth{\@biblabel{#1}}
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \setlength{\parsep}{1mm}
    \setlength{\labelsep}{0.5em}
    \setlength{\itemsep}{-\parsep}
    \setlength{\listparindent}{0in}
    \setlength{\itemindent}{0in}
    \setlength{\rightmargin}{0in}
    \@openbib@code
    \usecounter{enumiv}
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}
  }
  \sloppy
  \clubpenalty4000
  \@clubpenalty\clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m
}{
\def\@noitemerr
{\@latex@warning{Empty `thebibliography' environment}}
\endlist \vskip.2in}
\pretocmd{\bibliography}
  {\clearpage\phantomsection\addcontentsline{toc}{chapter}{\bibname}}
  {}{}
\bibpunct{[}{]}{,}{s}{}{,}
\renewcommand\NAT@citesuper[3]{\ifNAT@swa%
  \unskip\kern\p@\textsuperscript{\NAT@@open #1\NAT@@close}%
  \if*#3*\else\ (#3)\fi\else #1\fi\endgroup}
\DeclareRobustCommand\inlinecite{\@inlinecite}
\def\@inlinecite#1{\begingroup\let\@cite\NAT@citenum\citep{#1}\endgroup}
\let\onlinecite\inlinecite
\newenvironment{manref}{
  \begingroup\nuaa@font@bib
  \begin{list}{\@biblabel{\@arabic\c@enumiv}}{
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \setlength{\parsep}{1mm}
    \setlength{\labelsep}{0.5em}
    \setlength{\itemsep}{-\parsep}
    \setlength{\listparindent}{0in}
    \setlength{\itemindent}{0in}
    \setlength{\rightmargin}{0in}
    \usecounter{enumiv}
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}
  }
}{
  \end{list}
  \endgroup
}
\newcommand\mcite[1]{\textsuperscript{[\ref{#1}]}}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

\hypersetup{
  hidelinks,
  bookmarksnumbered=true,
}
\ifxetex
\RequirePackage{upgreek}
\sisetup{
  math-micro = {\upmu},
  text-micro = {\textmu},
}
\fi
\renewcommand\figureautorefname\figurename
\renewcommand\tableautorefname\tablename
\newcommand\subfigureautorefname\figureautorefname
\renewenvironment{proof}[1][\proofname]{
  \pushQED{\qed}%
  \normalfont
  \topsep0pt \partopsep0pt % no space before
  \trivlist
  \nuaafontparleft
  \item[\hskip\labelsep\hskip\parindent
        \nuaa@font@title\selectfont
    #1\@addpunct{:}]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}
\ifnuaa@lang@cn
  \renewenvironment{proof}[1][\proofname]{
    \pushQED{\qed}%
    \normalfont
    \topsep0pt \partopsep0pt % no space before
    \trivlist
    \nuaafontparleft
    \item[\hskip\labelsep\hskip\parindent
          \nuaa@font@title\selectfont
      #1\@addpunct{:}]\ignorespaces
  }{%
    \popQED\endtrivlist\@endpefalse
  }
  \newtheoremstyle{nuaaplain}
    {0pt}{0pt}
    {\nuaafontparleft}{\parindent}
    {}{：}
    {0em}
    {\nuaa@font@title\selectfont\thmname{#1}\thmnumber{#2}\thmnote{（#3）}}
\else\ifnuaa@lang@en
  \renewenvironment{proof}[1][\proofname]{
    \pushQED{\qed}%
    \normalfont
    \topsep0pt \partopsep0pt % no space before
    \trivlist
    \nuaafontparleft
    \item[\hskip\labelsep\hskip\parindent
          \itshape\selectfont
      #1\@addpunct{:}]\ignorespaces
  }{%
    \popQED\endtrivlist\@endpefalse
  }
  \newtheoremstyle{nuaaplain}
    {0pt}{0pt}
    {\nuaafontparleft\itshape\selectfont}{\parindent}
    {}{:~}
    {0em}
    {\nuaa@font@title\selectfont\thmname{#1}\thmnumber{ #2}\thmnote{ (#3)}}
\else\ifnuaa@lang@ja
  \renewenvironment{proof}[1][\proofname]{
    \pushQED{\qed}%
    \normalfont
    \topsep0pt \partopsep0pt % no space before
    \trivlist
    \nuaafontparleft
    \item[\hskip\labelsep\hskip\parindent
          \nuaa@font@title\selectfont
      #1\@addpunct{:}]\ignorespaces
  }{%
    \popQED\endtrivlist\@endpefalse
  }
  \newtheoremstyle{nuaaplain}
    {0pt}{0pt}
    {\nuaafontparleft}{\parindent}
    {}{: }
    {0em}
    {\nuaa@font@title\selectfont\thmname{#1}\thmnumber{#2}\thmnote{(#3)}}
\fi\fi\fi
\newcommand\nuaatheoremg[3][\@empty]{
  \newtheorem{#2}{#3}
  \expandafter\gdef\csname #2autorefname\endcsname{% 空格消除
    \expandafter\ifstrempty\expandafter{#1}{#3}{#1}}
}
\newcommand\nuaatheoremchap[3][\@empty]{
  \newtheorem{#2}{#3}[chapter]
  \expandafter\gdef\csname #2autorefname\endcsname{% 空格消除
    \expandafter\ifstrempty\expandafter{#1}{#3}{#1}}
}
\newcommand\nuaatheoremchapu[3][\@empty]{
  \newaliascnt{#2}{dummytheorem}
  \newtheorem{#2}[#2]{#3}
  \aliascntresetthe{#2}
  \expandafter\gdef\csname #2autorefname\endcsname{% 空格消除
    \expandafter\ifstrempty\expandafter{#1}{#3}{#1}}
}
\newtheorem{dummytheorem}{Dummy}[chapter]
\newcommand\nuaafontcaption{
  \ifnuaa@bachelor
    \nuaa@font@title\fontsize{10.5bp}{15.6bp}\selectfont
  \else
    \normalfont
  \fi
}
\captionsetup{labelsep=quad}
\captionsetup{justification=centerlast}
\DeclareCaptionFont{nuaacaption}{\nuaafontcaption}
\captionsetup{font=nuaacaption}
\captionsetup[subfigure]{font=nuaacaption}
\setlength\heavyrulewidth{0.5bp}
\setlength\lightrulewidth{0.5bp}
\setlength\cmidrulewidth{0.5bp}
\ifnuaa@bachelor
  \DeclareFloatFont{bachelor}{\linespread{1.5}\fontsize{10.5bp}{15.6bp}\selectfont}
  \floatsetup[table]{font=bachelor}
\fi
\setlist{nosep}
\setlist*{leftmargin=*}
\setlist[1]{labelindent=\dimexpr\parindent+\nuaaparleft\relax} %% Only the level 1
\newcommand{\nuaa@dateCn}{
  \zhdigits{\the\year}年\zhnumber{\the\month}月
}
\newcommand{\nuaa@dateEn}{
  \ifcase\the\month
  \or January%
  \or February%
  \or March%
  \or April%
  \or May%
  \or June%
  \or July%
  \or August%
  \or September%
  \or October%
  \or November%
  \or December%
  \fi, \the\year
}
\ifnuaa@blankleft
  \let\nuaa@cleardoublepage\cleardoublepage
  \renewcommand{\cleardoublepage}{
    \clearpage
    {
    \pagestyle{style@empty}
    \nuaa@cleardoublepage
    }
  }
\fi
\newbox\nuaa@kw
\newcommand{\nuaa@put@kw}[2]{%
  \begingroup
  \setbox\nuaa@kw=\hbox{#1}
  \noindent\hangindent\wd\nuaa@kw\hangafter1
  \box\nuaa@kw#2\par
  \endgroup}
\def\makecover{
  \ifnuaa@lang@cn
    \hypersetup{
      pdftitle = {\nuaa@value@title},
      pdfauthor = {\nuaa@value@author},
      pdfkeywords = {\nuaa@keywords@pdf}
    }
  \else\ifnuaa@lang@en
    \hypersetup{
      pdftitle = {\nuaa@valueEn@title},
      pdfauthor = {\nuaa@valueEn@author},
      pdfkeywords = {\nuaa@keywordsEn@pdf}
    }
  \else\ifnuaa@lang@ja
    \hypersetup{
      pdftitle = {\nuaa@valueJa@title},
      pdfauthor = {\nuaa@value@author},
      pdfkeywords = {\nuaa@keywordsJa@pdf}
    }
  \fi\fi\fi
  \pagestyle{style@empty}
  \pagenumbering{Alph}
  \cleardoublepage

  \ifnuaa@bachelor
    \nuaa@make@cover@bachelor
  \else
    \nuaa@make@cover@master@cn
    \nuaa@make@cover@master@en
  \fi
}
\newcommand\makedeclare{
  \ifnuaa@bachelor
    \nuaa@make@declare@bachelor
  \else
    \nuaa@make@declare@master
  \fi
}
\newcommand\makeabstract{
  \cleardoublepage

  \ifnuaa@bachelor
    \nuaa@make@abstract@bachelor@cn
    \ifnuaa@lang@ja
      \nuaa@make@abstract@bachelor@ja
    \else
      \nuaa@make@abstract@bachelor@en
    \fi
  \else
    \nuaa@make@abstract@master@cn
    \ifnuaa@lang@ja
      \nuaa@make@abstract@master@ja
    \else
      \nuaa@make@abstract@master@en
    \fi
  \fi
}
\newcommand\nuaatableofcontents{
  \cleardoublepage
  \chapter*{
    \ifnuaa@bachelor
      \linespread{1.5}\fontsize{16bp}{15.6bp}\selectfont
    \fi
    \contentsname}
  \@starttoc{toc}
}
\newcommand\nuaalistoffigurestables{
  \clearpage
  \chapter*{\listfiguretablename}
  \@starttoc{lof}
  \bigskip
  \@starttoc{lot}
}
\newcommand\nuaa@make@cover@bachelor{
  \cleardoublepage
  \newgeometry{top=1.0in, bottom=1.0in, left=1.25in, right=1.25in}

  \begin{flushright}
    \linespread{1.25}\sffamily\heiti\fontsize{14bp}{16.8bp}\selectfont
    \nuaa@label@thesisnum\underline{\hspace{60bp}}
    \vspace{\stretch{4}}
  \end{flushright}

  \begin{center}
    \linespread{1}\kaishu\fontsize{22bp}{30bp}\selectfont
    \nuaa@textbf{\nuaa@university}
    \vspace{\stretch{3}}

    \linespread{1}\songti\fontsize{55bp}{55bp}\selectfont
    \nuaa@textbf{\nuaa@worktypecn}
    \vspace{\stretch{5}}
  \end{center}

  \begin{center}
    \linespread{1.212}\fontsize{22bp}{26.4bp}\selectfont
    \begin{tabular}{C{1.4in}C{4.0in}}
    {\sffamily\heiti\nuaa@label@title} &
    \ifnuaa@lang@ja
      {\sffamily\gtfamily\nuaa@valueJa@title \par}
    \else\ifnuaa@lang@en
      {\rmfamily\nuaa@valueEn@title \par}
    \fi\fi
    {\sffamily\heiti\nuaa@value@title}
    \end{tabular}
    \vspace{\stretch{5}}

    \linespread{1}\sffamily\heiti\fontsize{16bp}{47.6bp}\selectfont
    \begin{tabular} {cc}
      \nuaa@label@author & \nuaa@value@author \\ \cline{2-2}
      \nuaa@label@studentid & \nuaa@value@studentid \\ \cline{2-2}
      \nuaa@label@college & \nuaa@value@college \\ \cline{2-2}
      \nuaa@label@major & \nuaa@value@major \\ \cline{2-2}
      \nuaa@label@classid & \makebox[3.22in]{\nuaa@value@classid} \\ \cline{2-2}
      \nuaa@label@adviser & \nuaa@value@advisers \\ \cline{2-2}
    \end{tabular}
    \vspace{\stretch{3}}

    \linespread{1}\sffamily\heiti\fontsize{16bp}{30bp}
    \ifdefempty{\nuaa@value@applydate}{\nuaa@dateCn}{\nuaa@value@applydate}
  \end{center}
  \restoregeometry
}
\newcommand\nuaa@make@declare@bachelor{
  \cleardoublepage
  \newgeometry{top=1.0in, bottom=1.0in, left=1.25in, right=1.25in}

  \begin{center}
    \linespread{1.0}\heiti\fontsize{18bp}{31.2bp}\selectfont
    \nuaa@textbf{\nuaa@university} \par
    \nuaa@textbf{本科\nuaa@worktypecn 诚信承诺书}
  \end{center}

  \begingroup
    \linespread{1.0}\songti\fontsize{14bp}{31.2bp}\selectfont
    \setlength\parindent{2\ccwd}\indent

    本人郑重声明：所呈交的\nuaa@worktypecn
    （题目：{%
      \renewcommand\linebreak{}%
      \ifnuaa@lang@cn{\expandafter\uline\expandafter{\nuaa@value@title}}\else%
      \ifnuaa@lang@en{\expandafter\uline\expandafter{\nuaa@valueEn@title}}\else%
      \ifnuaa@lang@ja{\mcfamily\expandafter\uline\expandafter{\nuaa@valueJa@title}}%
      \fi\fi\fi%
    }）
    是本人在导师的指导下独立进行研究所取得的成果。
    尽本人所知，除了\nuaa@worktypecn 中特别加以标注引用的内容外，
    本\nuaa@worktypecn 不包含任何其他个人或集体已经发表或撰写的成果作品。

    \vspace{31.2bp}

    \begin{flushright}
      \setlength{\tabcolsep}{0bp}
      \begin{tabular}{rcr}
      作者签名： & \hspace{7.5em} & \hspace{2em} 年 \hspace{0.75em} 月 \hspace{0.75em} 日 \\
      （学号）： & \hspace{7.5em} & \\
      \end{tabular}
    \end{flushright}

  \endgroup
  \restoregeometry
}
\newcommand\nuaa@make@abstract@bachelor@cn{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \begin{center}
    \vspace*{-4.3pt}\sffamily\heiti\zihao{2}
    \phantomsection
    \addcontentsline{toc}{chapter}{\nuaa@label@abstractshort}
    \nuaa@value@title
  \end{center}

  \begin{center}
    \sffamily\heiti\zihao{-3}\vspace{1em}
    \nuaa@label@abstract
  \end{center}

  \begingroup
    \setlength\parindent{2\ccwd}\songti\indent

    \nuaa@abstract
  \endgroup

  \vspace{3em}

  \nuaa@put@kw{\heiti\zihao{-3}\nuaa@label@keywords}{\songti\nuaa@keywords}
}
\newcommand\nuaa@make@abstract@bachelor@en{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \begin{center}
    \vspace*{-4.3pt}\sffamily\heiti\zihao{2}
    \phantomsection
    \addcontentsline{toc}{chapter}{\nuaa@labelEn@abstract}
    \nuaa@valueEn@title
  \end{center}

  \begin{center}
    \sffamily\heiti\zihao{-3}\vspace{18pt}
    \nuaa@labelEn@abstract
    \vspace{10pt}
  \end{center}

  \begingroup
    \setlength\parindent{1em}\rmfamily\songti
    \nuaa@abstractEn
  \endgroup

  \vspace{3em}

  \nuaa@put@kw{\zihao{-3}\nuaa@labelEn@KeyWords}{\nuaa@keywordsEn}
}
\newcommand\nuaa@make@abstract@bachelor@ja{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \begin{center}
    \vspace*{-4.3pt}\sffamily\gtfamily\zihao{2}
    \phantomsection
    \addcontentsline{toc}{chapter}{\nuaa@labelJa@abstract}
    \nuaa@valueJa@title
  \end{center}

  \begin{center}
    \sffamily\gtfamily\zihao{-3}\vspace{18pt}
    \nuaa@labelJa@abstract
    \vspace{10pt}
  \end{center}

  \begingroup
    \setlength\parindent{1em}\rmfamily\mcfamily
    \nuaa@abstractJa
  \endgroup

  \vspace{3em}

  \nuaa@put@kw{\nuaa@textbf{\zihao{-3}\nuaa@labelJa@keywords}}{\nuaa@keywordsJa}
}
\newcommand\nuaa@make@cover@master@cn{
  \cleardoublepage

  \begin{multicols}{2}
    \linespread{1}\songti\fontsize{10.5bp}{15.6bp}\selectfont
    \begin{flushleft}
      中图分类号：\nuaa@value@libraryclassid \par
      学科分类号：\nuaa@value@subjectclassid
    \end{flushleft}
    \columnbreak
    \begin{flushright}
      论文编号：\nuaa@value@thesisid
    \end{flushright}
  \end{multicols}
  \vspace{\stretch{2}}

  \begin{center}
    \linespread{1}\songti\fontsize{42bp}{62.4bp}\selectfont\nuaa@worktypecn
    \vspace{\stretch{2}}

    \linespread{1}\sffamily\heiti\fontsize{26bp}{46.8bp}\selectfont\nuaa@value@title
  \end{center}
  \vspace{\stretch{3}}

  \begin{center}
  \linespread{1}\rmfamily\songti\fontsize{16bp}{31.2bp}\selectfont
  \renewcommand\linebreak{\par}
  \begin{tabular} {@{\hskip 1.125in}c@{\hskip .6389in}p{3.1115in}}
    \nuaa@label@researchername & \nuaa@value@author \\
\ifnuaa@zhuanshuo
    \nuaa@label@professionaltype & \nuaa@value@majorsubject \\
    \nuaa@label@professionalfield & \nuaa@value@researchfield \\
\else
    \nuaa@label@majorsubject & \nuaa@value@majorsubject \\
    \nuaa@label@researchfield & \nuaa@value@researchfield \\
\fi
    \nuaa@label@adviser & \nuaa@value@advisers \\
  \end{tabular}
  \end{center}
  \vspace{\stretch{3}}

  \begin{center}
    \linespread{1}
    \includegraphics{nuaa-jianqi.pdf}

    \kaishu\fontsize{18bp}{31.2bp}\selectfont
    \nuaa@label@graduateschool\quad \nuaa@value@college

    \kaishu\fontsize{16bp}{31.2bp}\selectfont
    \ifdefempty{\nuaa@value@applydate}{\nuaa@dateCn}{\nuaa@value@applydate}

  \end{center}
}
\newcommand\nuaa@make@cover@master@en{
  \cleardoublepage

  \begin{center}
    \linespread{1.5}\fontsize{14bp}{14bp}\rmfamily\selectfont
    \nuaa@labelEn@nuaa \par
    \nuaa@labelEn@graduateschool \par
    \nuaa@valueEn@college \vspace{\stretch{1}}

    \linespread{1}\fontsize{22bp}{31.2bp}\rmfamily\selectfont
    \ifnuaa@lang@ja
      \jpn{\nuaa@textbf{\nuaa@valueJa@title}}
    \else
      \nuaa@textbf{\nuaa@valueEn@title}
    \fi \par \vspace{\stretch{1}}

    \linespread{1.75}\fontsize{14bp}{16.8bp}\rmfamily\selectfont
    A Thesis in \\
    \nuaa@valueEn@majorsubject \\
    by \\
    \nuaa@valueEn@author \\
    Advised by \\
    \nuaa@valueEn@advisers \par \bigskip

    Submitted in Partial Fullfillment \\
    of the Requirements \\
    for the Degree of \\
    \nuaa@valueEn@degreefull \par\bigskip

    \ifdefempty{\nuaa@valueEn@applydate}{\nuaa@dateEn}{\nuaa@valueEn@applydate}
  \end{center}
}
\newcommand\nuaa@make@declare@master{
  \cleardoublepage

  \begin{center}
  \linespread{1.0}\songti\fontsize{22bp}{62.4bp}\selectfont
  \vspace*{25.5bp} \vspace{-\parskip}\vspace{-\baselineskip}
  承诺书 \par
  \end{center}

  \begingroup
  \vspace*{31.4bp} \vspace{-\parskip}\vspace{-\baselineskip}
  \linespread{1.0}\rmfamily\songti\fontsize{16bp}{30bp}\selectfont
  \setlength\parindent{2\ccwd}

  本人声明所呈交的\nuaa@worktypecn 是本人在导师指导下进行的研究工作及取得的研究成果。
  除了文中特别加以标注和致谢的地方外，论文中不包含其他人已经发表或撰写过的研究成果，
  也不包含为获得\nuaa@label@nuaa 或其他教育机构的学位或证书而使用过的材料。

  本人授权\nuaa@label@nuaa 可以将学位论文的全部或部分内容编入有关数据库进行检索，
  可以采用影印、缩印或扫描等复制手段保存、汇编学位论文。

  （保密的学位论文在解密后适用本承诺书）

  \endgroup

  \vfill
  \begin{flushright}
  \linespread{1.0}\songti\fontsize{14bp}{25bp}\selectfont
  \makebox[5\ccwd][c]{作者签名：} \underline{\hspace{7em}} \par
  \makebox[5\ccwd][c]{日 \hfill 期：} \underline{\hspace{7em}} \par
  \end{flushright}
  \vfill
}
\newcommand\nuaa@make@abstract@master@cn{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \chapter*{\heiti\nuaa@label@abstract}

  \begingroup
    \setlength\parindent{2\ccwd}
    \rmfamily\songti\indent

    \nuaa@abstract
  \endgroup
  \vskip 2\baselineskip minus 1.5\baselineskip

  \nuaa@put@kw{\nuaa@textbf{\songti\nuaa@label@keywords}}{\songti\nuaa@keywords}
}
\newcommand\nuaa@make@abstract@master@en{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \chapter*{\textrm{\nuaa@textbf{\nuaa@labelEn@ABSTRACT}}}

  \begingroup
    \setlength\parindent{1em}
    \nuaa@abstractEn
  \endgroup
  \vskip 2\baselineskip minus 1.5\baselineskip

  \nuaa@put@kw{\nuaa@textbf{\nuaa@labelEn@keywords}}{\nuaa@keywordsEn}
}
\newcommand\nuaa@make@abstract@master@ja{
  \ifnuaa@abstractopenright
    \cleardoublepage
  \else
    \clearpage
  \fi

  \chapter*{\nuaa@textbf{\mcfamily\nuaa@labelJa@abstract}}

  \begingroup
    \nuaa@abstractJa
  \endgroup
  \vskip 2\baselineskip minus 1.5\baselineskip

  \nuaa@put@kw{\nuaa@textbf{\nuaa@labelJa@keywords}}{\nuaa@keywordsJa}
}

\newcommand{\nuaathesis}{%
  \makebox{\rmfamily%
    N\hspace{-0.3ex}\raisebox{-0.5ex}{U}\hspace{-0.3ex}A\mbox{\textsuperscript{\hspace{-0.5ex}2}}\hspace{0.3ex}%
    \textsc{Thesis}}}
\newcommand{\oldnuaathesis}{%
  N\raisebox{0.5ex}{U}\hspace{-0.3ex}AA%
  \textsc{Thesis}
}
\newcommand{\seuthesix}{%
  \makebox{S\hspace{-0.3ex}\raisebox{-0.5ex}{E}\hspace{-0.3ex}U\hspace{0.1em}%
  \textsc{Thesix}}
}
\newcommand\cquthesis{\textsc{Cqu}\-\textsc{Thesis}}
%% 
%% Copyright (C) 2018 by nuaatug
%% 
%% Licensed under the Apache License, Version 2.0 (the "License");
%% you may not use this file except in compliance with the License.
%% You may obtain a copy of the License at
%% 
%%     http://www.apache.org/licenses/LICENSE-2.0
%% 
%% Unless required by applicable law or agreed to in writing, software
%% distributed under the License is distributed on an "AS IS" BASIS,
%% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%% See the License for the specific language governing permissions and
%% limitations under the License.
%% 
%% This work consists of the file  nuaathesis.dtx
%% and the derived files           nuaathesis.ins,
%%                                 nuaathesis.cls,
%%                                 nuaathesis.cfg and
%%                                 nuaathesis.pdf
%% 
%%
%% End of file `nuaathesis.cls'.
