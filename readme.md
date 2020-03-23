2020-03-22

# Readme

*Titre :*
Détection de zones par montée de gradient d'un rectangle dans le flou gaussien.

*Contexte :*
On considère un nuage de *m* points dans un espace de dimension *n*.
Chaque point est à valeurs binaires {0,1}.

*But :*
Déterminer des hypercubes dans lesquels la proportion locale de 0 et de 1 est statistiquement éloignée de la proportion globale de 0 et de 1.

*Idée :*

1. On considère le plus petit hypercube contenant l'ensemble des *m* points.
2. On redimensionne ce dernier hypercube à un hypercube normalisé *[0,1]ⁿ*.
3. On classifie les points dans des cases pour une résolution donnée. Ceci donne lieu à une fonction de proportion locale, qu'on ajuste selon la proportion globale.
4. On applique un noyau gaussien pour lisser cette dernière fonction.
5. On fait une montée de gradient d'un rectangle dans cette fonction lissée.
6. On regarde où le rectangle a convergé.

*Remarques :*
Pour l'instant, le présent code Python 3 :

1. ne fonctionne qu'en dimension *n=2*,
2. ne regarde pas si ce qu'il a trouvé est statistiquement intéressant ou non.