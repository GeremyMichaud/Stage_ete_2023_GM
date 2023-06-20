%% Thesis.cls
%% Copyright 2023 G. Michaud
%
% This model is inspired by that of Yannick Lapointe.
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   http://www.latex-project.org/lppl.txt
% and version 1.3 or later is part of all distributions of LaTeX
% version 2005/12/01 or later.
%
% This work has the LPPL maintenance status `maintained'.
% 
% The Current Maintainer of this work is G. Michaud.
%
% This work consists of the file Thesis.cls.


% ---------------
% Classe custom pour les devoirs.
% Par Gérémy "Ken" Michaud
% ---------------

\ProvidesClass{Thesis}[2023/06/08 v0.1 Thesis Class]
\LoadClass[11pt]{article}

% ---------------
% Options de la classe
% ---------------

\usepackage{pifont}
\usepackage{setspace}

\newif\ifIcomma                     % Option pour retirer icomma du ProjectPackage
\DeclareOption{comma}{\Icommatrue}
\DeclareOption{dot}{\Icommafalse}
\ExecuteOptions{comma}
\ProcessOptions*

\newif\ifCompact                    % Option pour "page titre" compacte
\DeclareOption{compact}{\Compacttrue}
\DeclareOption{fullpage}{\Compactfalse}
\ExecuteOptions{fullpage}
\ProcessOptions*

% ---------------
% Packages et présentation générale
% ---------------

\ifIcomma
    \RequirePackage[comma]{ProjectPackage}
\else
    \RequirePackage[dot]{ProjectPackage}
    \ClassWarningNoLine{homework}{Package icomma not imported by default. To import use option "comma"}
\fi

\ifCompact
    \geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\fi

\mdfsetup{
    backgroundcolor=gray!10!white,
    splittopskip=2\topsep,
    nobreak=true
}

\fancypagestyle{firstpage}{
    \fancyhf{}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

\renewcommand{\theequation}{\arabic{equation}}
\setlength{\headheight}{13.6pt}

\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}}
\renewcommand{\thesubsubsection}{(\alph{subsection}.\roman{subsubsection})}

\pagestyle{fancy}
\parindent=0pt                      % Indentation des paragraphes
\parskip=6pt                        % Espace entre les paragraphes
\fancyheadoffset{0cm}               % Largeur en-tête
\rhead{
    \@ifundefined{@shorttitle}{\bfseries \@title}{
        {\bfseries \@shorttitle}}
}
\lhead{                             % Versions courtes des noms en haut à gauche
    \@ifundefined{@shortone}{}{
        \@shortone\!\!}
    \@ifundefined{@shorttwo}{}{
        \!\!, \@shorttwo\!\!}
    \@ifundefined{@shortthree}{}{
        \!\!, \@shortthree\!\!}
    \@ifundefined{@shortfour}{}{
        \!\!, \@shortfour\!\!}
    \@ifundefined{@shortfive}{}{
        \!\!, \@shortfive\!\!}
}
\cfoot{\thepage}                    % Pagination en pied de page

% ---------------
% Commandes customs pour la page titre
% ---------------

\def \@title {Titre du devoir}      % Titre par défaut si aucun n'est spécifié
\def \@date {\today}                % Si aucune date n'est spécifiée
\def \title #1{\def \@title {#1}}
\def \shorttitle #1 {\def \@shorttitle {#1}}

\def \author #1#2{              % Jusqu'à 5 auteurs et la version courte de leur nom peuvent être spécifiés
    \@ifundefined{@authorone}{
        \def \@authorone {#1}
        \def \@shortone {#2}
    }{
    \@ifundefined{@authortwo}{
        \def \@authortwo {#1}
        \def \@shorttwo {#2}
    }{
    \@ifundefined{@authorthree}{
        \def \@authorthree {#1}
        \def \@shortthree {#2}
    }{
    \@ifundefined{@authorfour}{
        \def \@authorfour {#1}
        \def \@shortfour {#2}
    }{
        \def \@authorfive {#1}
        \def \@shortfive {#2}
    }}}}
}

\def \teacher #1{                   % Jusqu'à 3 professeurs peuvent être spécifiés
    \@ifundefined{@teacherone}{
        \def \@teacherone {#1}
    }{
    \@ifundefined{@teachertwo}{
        \def \@teachertwo {#1}
    }{
        \def \@teacherthree {#1}
    }}
}

\def \printauthor {               % Permet d'afficher les auteurs
    \@ifundefined{@authorone}{}{
        \@authorone}
    \@ifundefined{@authortwo}{}{\\
        \@authortwo}
    \@ifundefined{@authorthree}{}{\\
        \@authorthree}
    \@ifundefined{@authorfour}{}{\\
        \@authorfour}
    \@ifundefined{@authorfive}{}{\\
        \@authorfive}
}

\def \printteacher {            % Permet d'afficher les profs
    \@ifundefined{@teacherone}{}{
        \@teacherone}
    \@ifundefined{@teachertwo}{}{\\
        \@teachertwo}
    \@ifundefined{@teacherthree}{}{\\
        \@teacherthree}
}

\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

% ---------------
% Formattage de la page titre automatique avec le \begin{document}
% ---------------

\ifCompact
    \let\ori@document=\document
    \renewcommand{\document}{
        \ori@document
        \thispagestyle{firstpage}
        \begin{center}
            \begin{minipage}{5cm}
                \includegraphics[width=10cm]{Logo_Ulaval.png}
            \end{minipage}
            \begin{minipage}{11cm}
                \begin{flushright}
                    \@date\\[0.2cm]
                \end{flushright}
            \end{minipage}\\[0.8cm]
            \begin{center}
                \huge\bfseries\@title
            \end{center}
            \begin{minipage}{0.49\textwidth}
                \emph{Par :}\\
                \printauthor
            \end{minipage}
            \begin{minipage}{0.49\textwidth}
                \begin{flushright}
                    \emph{Remis à :}\\
                    \printteacher
                \end{flushright}
            \end{minipage}\\[0.4cm]
        \end{center}
        \HRule
    }
\else
    \let\ori@document=\document
    \renewcommand{\document}{
        \ori@document
        \thispagestyle{empty}
        \begin{titlepage}
            \center
            \quad\\
            \includegraphics[width=13cm]{Logo_Ulaval.png}\\
            \includegraphics[width=8cm]{Logo_CHUQ.png}\\[2cm]
            \HRule \\[0.4cm]
            {\huge\bfseries \@title \par}
            \HRule\\[1.5cm]
            \begin{minipage}{0.4\textwidth}
                \begin{flushleft}\large
                    \emph{Par :}\\
                    \printauthor
                \end{flushleft}
            \end{minipage}
            \begin{minipage}{0.4\textwidth}
                \begin{flushright}\large
                    \emph{Sous la direction de :} \\
                    \printteacher
                \end{flushright}
            \end{minipage}\\\vspace*{\fill}
            {\large \@date}
        \end{titlepage}
    }
\fi