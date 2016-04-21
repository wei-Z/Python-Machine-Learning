# Introduction
s = "Regular expressions  easily explained!"
"easily" in s

# Syntax of Regular Expressions
import re
x = re.search("cat", "A cat and a rat can't be friends.")
print x
x = re.search("cow", "A cat and a rat can't be friends.")
print x


if re.search("cat", "A cat  and a rat can't be friends."):
    print "Some kind of cat has been found :-"
else:
    print "Some kind of cat has been found :-"

if re.search("cow", "A cat and a rat can't be friends."):
    print "Cats and Rats and a cow."
else:
    print "No cow around."

# Any Character
# Character Classes
"""
Square brackets, "[" and"]", are used to included a character class.
[xyz] means e.g. either an "x", and "y" or a "z". Let's look at a more
practical example:
r"M[ae][iy]er"
This is a regular expression, which matches a surname which is quite
common in German. A name with the same pronunciation and four
different spellings: Maier, Mayer, Meier, Meyer
A finite state automate to recognize this expression can be built like this:

Instead of a choice between two characters, we often need a choice
between larger character classes. Like a class of letters between "a"
and "e" or between "0" and "5".
To manage such character classes the syntax of regular expressions
supplies a metacharacter "-". [a-e] a simplified writing for [abcde] or
[0-5] denotes [012345].
"any uppercase letter" [A-Z].
"any lower case or uppercase letter" [A-Za-z].
There is something more about the dash, we used to mark the begin
and the end of a character class. The dash has only a special meaning
if it is used within square brackets and in this case only if it isn't positioned
directly after an opening or immediately in front of a closing bracket.
So the expression [-az is only the choice between the three characters "-",
"a" and "z", but no other characters. The same is true for [az-.

Exercise:
What character class is described by [-a-z]?
Answer:
The character "-" and all the characters "a", "b", "c" all the way up to "z".

The only other special character inside quare brackets (chatacter class
choice) is the catet "^". If it is used directly after an opening square bracket,
it negates the choice. [^0-9] denotes the choice " any character but a digit".
The position of the caret within the square brackets is curcial. If it is not
positioned as the first character following the opening square bracket, it has
no special meaning.
[^abc] means anything but an "a", "b" or "c"
[a^bc] means an "a", "b", "c" or a "^".

"""
# A Practical Exercise  in Python
import re

fh = open("simpsons_phone_book.txt")
for line in fh:
    if re.search(r"J.*Neu", line):
        print line.rstrip()
fh.close()

"""
These are all lower case and uppercase characters plus all the digits
and the underscore, corresponding to the following regular expression:
r"[a-zA-Z0-9]"

The special sequences consist of "\\" and a character from the following:

\d  Matches any decimal digit; equivalent to the set [0-9].

\D  The complement of \d. It matches any non-digit character; equivalent to
       the set [^0-9]

\s   Matches any whitespace character; equivalent to [\t\n\r\f\v].

\S   The complement of \s. It matches any non-whitespace character;
       equiv. to [^ \t\n\r\f\v].

\w   [a-zA-Z0-9]. With LOCALE, it will match the set [a-zA-Z0-9] plus
       characters defined as letters for the current locale.

\W   Matches the complement of \w.

\b     Matches the empty string, but only at the start or end of a word.

\B    Matches the empty string, but not at the start or end of a word.

\\      Matches a literal backslash.


The \b and \B of the previous overview of special sequences, is often
not properly understood or even misunderstood especially by novices.
While the other sequences match characters, - e.g. \w matches characters
like "a", "b", "m", "3" and so on, - \b and \B don't match a character.
They match empty strings depending on their neighbourhood, i.e.
what kind of a character the predecessor and the successor is. So \b
matches any empty string between a \W and a \w character and also
between a \w and a \W character. \B is the complement, i.e empty strings
between \W and \W or empty strings between \w and \w. We illustrate this i
n the following example:

"""

# http://www.python-course.eu/re.php
# -*- coding: utf-8 -*-
# Mactching Beginning and End
# As we have carried out previously in this introduction, the expression r"M[ae][iy]er" is capable of matching
# various spellings of the name Mayer and the name can be anywhere in the string.
import re
line = "He is a German called Mayer"
if re.search(r"M[ae][iy]er", line): print "I foundone!"

"""
But what if we want to match a regular expression at the beginning of a string and only at the beginning?
The re module of Python provides two functions to match regular expression. We have met already one of
them, i.e. search(). The other has in our opinion a misleading name: match()
Misleading, because match(re_str, s) checks for a match of re_str merely at the beginning of the string.
But anyway, match() is the solution to our question, as we can see in the following example:

"""
import re
s1 = "Mayer is a very common Name"
s2 = "He is called Meyer but he isn't German."
print re.search(r"M[ae][iy]er", s1)
print re.search(r"M[ae][iy]er", s2)
print re.match(r"M[ae][iy]er", s1)
print re.match(r"M[ae][iy]er", s2)

"""
The caret '^' Matches the start of the string, and in MULILIEN (will be explained further down) mode also matches
immediately after each newline, which the Python method match() doesn't do.

"""
import re
s1 = "Mayer is a very common Name"
s2 = "He is called Meyer but he isn't German."
print re.search(r"^M[ae][iy]er", s1)
print re.search(r"^M[ae][iy]er", s2)

# But what happens if we concatenate the two strings s1 and s2 in the following way:
# s = s2 + "\n" + s1
# Now the string doesn't start with a Maier of any kind, but the name is following a newline character:
s = s2 + "\n" + s1
print re.search(r"^M[ae][iy]er", s)

# The name hasn't been found, becauseonly the beginning of the string is checked. It changes, if we use
# the multiline mode, which can be activated by adding the following third parameters to search:
print re.search(r"^M[ae][iy]er", s, re.MULTILINE)
print re.search(r"^M[ae][iy]er", s, re.M)
print re.match(r"^M[ae][iy]er", s, re.M)

print re.search(r"Python\.$", "I like Python.") #<_sre.SRE_Match object at 0x0000000008592030>
print re.search(r"Python\s$", "I like Python and Perl.") #None
print re.search(r"Python\.$", "I like Python.\nSome prefer Java or Perl.") #None
print re.search(r"Python\.$", "I like Python.\nSome prefer Java or Perl.", re.M) #<_sre.SRE_Match object at 0x0000000008592030>

# Optional Items
"""

If you thought that our collection of Mayer names was complete, you were wrong.
There are other ones all over the world, e.g. London and Paris, who dropped their "e".
So we have four more names ["Mayr", "Meyr", "Meir", "Mair"] plus our old set ["Mayr", "Meyr", "Meir", "Mair"].

If we try to figure out a fitting regular expression, we realize that we miss something.
A way to tell the computer "this "e" may or may not occur". A question mark is used as a notation for this.
A question mark declares that the preceding character or expression is optional.

The final Mayer-Recognizer looks now like this:

"""
r"M[ae][iy]e?r"

#A subexpression is grouped by round brackets and a question mark following such a group means that this group
#may or may not exist. With the following expression we can match dates like "Feb 2011" or February 2011":
r"Feb(ruary)? 2011"

# Quantifiers
"""

If you just use, what we have introduced so far, you will still need a lot of things.
Above all some way of repeating characters or regular expressions.
We have already encountered one kind of repetition, i.e. the question mark.
A character or a subexpression is "repeated" not at all or exactly on time.

We have also used previously another quantifier without explaining it.
It is the asterisk or star character. A star following a character or a subexpression means that this expression
or character may be repeated arbitrarily, even zero times.

"""
r"[0-9]*"
# The above expression matches any sequence of digits, even the empty string.
# r".*" matches any sequence of characters and the empty string.
# Exercise
# Write a gegular expression which matches strings which starts with a swquence of digits - at least one digit
# - followed by a blank and after this arbitrary characters.
# Solution:
r"^[0-9][0-9] .*"

# Solution with plus quantifier:
r"^[0-9]+ .*"

"""
If you work for a while with this arsenal of operators, you will miss inevitably at some point the possibility to
repeat expressions for an exact number of times. Let's assume you want to recognize the last lines of addresses
on envelopes in Switzerland. These lines usually contain a four digits long post code followed by a blank and
a city name. Using + or * are too unspecific for our purpose and the following expression seems to be too
clumsy:

"""
r"^[0-9][0-9][0-9][0-9] [A-Za-z]+"

# Fortunately, there is an alternative available:
r"[0-9]{4} [A-Za-z]*"

"""
Now we want to improve our regular expression. Let's assume that there is no city name in Switzerland,
which consists of less than 3 letters, at least 3 letters. We can denote this by [A-Za-z]{3,}.
Now we have to recognize lines with German post code (5 digits) lines as well, i.e. the post code can now
consist of either four or five digits:

"""
r"^[0-9]{4,5} [A-Z][a-z]{2,}"
#The general syntax is {from, to}: this means that the expression has to appear at least "from" times
#and not more than "to" times. {, to} is an abbreviated spelling for {0,to} and {from,} is an abbreviation for
#"at least from times but no upper limit"

# Grouping
"""
We can group a part of a regular expression by surrounding it with parenthesis (round brackets).
This way we can apply operators to the complete group instead of a single character.
"""

#Capturing Groups and Backreferences
"""
Parenthesis (round brackets) not only group subexpressions but they create backreferences as well.
The part of the string matched by the grouped part of the regular expression, i.e. the subexpression
in parenthesis, is stored in a backreference. With the aid of backreferences we can reuse parts of regular
expressions. These stored values can be both reused inside the expression itself and afterwards, when
the regexpr will have been executed. Before we continue with our treatise about backreferences, we want
to strew in a paragraph about match objects, which is important for our next examples with backreferences.

"""

# A Closer Look at the Match Object
"""
A match object contains the methods group(), span(), start() and end(), as can be seen in the following application:
"""
import re
mo = re.search("[0-9]+", "Customer number: 232454, Date: February 12, 2011")
mo.group() # '232454'
mo.span() # (17, 23)
mo.start() # 17
mo.end() # 23
mo.span()[0] # 17
mo.span()[1] # 23

"""
These methods are not difficult to understand. span() returns a tuple with the start and end position,
i.e. the string index where the regular expression started matching in the string and ended matching.
The methods start() and end() are in a way superfluous as the information is contained in span(),
i.e. span()[0] is equal to start() and span()[1] is equal to end(). group(), if called without argument,
returns the substring, which had been matched by the complete regular expression. With the help of
group() we are also capable of accessing the matched substring by grouping parentheses, to get the
matched substring of the n-th group, we call group() with the argument n: group(n).
We can also call group with more than integer argument, e.g. group(n,m). group(n,m) - provided there
exists a subgoup n and m - returns a tuple with the matched substrings. group(n,m) is equal to (group(n),
group(m)):
"""
import re
mo = re.search("([0-9]+).*: (.*)", "Customer number: 232454, Date: February 12, 2011")
mo.group() # '232454, Date: February 12, 2011'
mo.group(1)  # '232454'
mo.group(2) # 'February 12, 2011'
mo.group(1, 2) # ('232454', 'February 12, 2011')

"""
A very intuitive example are XML or HTML tags. E.g. let's assume we have a file (called "tags.txt") with content like
this:

<composer>Wolfgang Amadeus Mozart</composer>
<author>Samuel Beckett</author>
<city>London</city>

We want to rewrite this text automatically to

composer: Wolfgang Amadeus Mozart
author: Samuel Beckett
city: London
"""

"""
The following little Python script does the trick. The core of this script is the regular expression.
This regular expression works like this: It tries to match a less than symbol "<".
After this it is reading lower case letters until it reaches the greater than symbol.
Everything encountered within "<" and ">" has been stored in a backreference which can be accessed
within the expression by writing \1. Let's assume \1 contains the value "composer": When the expression
has reached the first ">", it continues matching, as the original expression had been
"<composer>(.*)</composer>":
"""
import re
#fh = open("tags.txt")
fh = ["<composer>Wolfgang Amadeus Mozart</composer>",  \
"<author>Samuel Beckett</author>", \
"<city>London</city>"]
for i in fh:
    res = re.search(r"<([a-z]+)>(.*)</\1>", i)
    print res.group(1) + ": " + res.group(2)
"""
If there are more than one pair of parenthesis (round brackets) inside the expression,
the backreferences are numbered \1, \2, \3, in the order of the pairs of parenthesis.
"""

"""
Exercise:
The next Python example makes use of three backreferences. We have an imaginary phone list of the Simpsons
in a list. Not all entries contain a phone number, but if a phone number exists it is the first part of an entry.
Then follows separated by a blank a surname, which is followed by first names. Surname and first name are
separated by a comma. The task is to rewrite this example in the following way:

Allison Neu 555-8396
C. Montgomery Burns
Lionel Putz 555-5299
Homer Jay Simpson 555-7334

"""
# Python script solving the rearrangement problem:
import re

l = ["555-8396 Neu, Allison",
     "Burns, C. Montgomery",
     "555-5299 Putz, Lionel",
     "555-7334 Simpson, Homer Jay"]

for i in l:
    res = re.search(r"([0-9-]*)\s*([A-Za-z]+),\s+(.*)", i)
    #print res.group()
    print res.group(3) + " " + res.group(2) + " " + res.group(1)

# Named Backreferences
"""
In the previous paragraph we introduced "Capturing Groups" and "Backreferences".
More precisely, we could have called them "Numbered Capturing Groups" and "Numbered Backreferences".
Using capturing groups instead of "numbered" capturing groups allows you to assign descriptive names
instead of automatic numbers to the groups. In the following example, we demonstrate this approach by
catching the hours, minutes and seconds from a UNIX date string.

"""
import re
s = "Sun Oct 14 13:47:03 CEST 2012"
expr = r"\b(?P<hours>\d\d):(?P<minutes>\d\d):(?P<seconds>\d\d)\b"
x = re.search(expr, s)
x.group('hours')
x.group('minutes')
x.start('minutes')
x.end('minutes')
x.span('seconds')

# Comprehensive Python Exercise
"""
In this comprehensive exercise, we have to bring together the information of two files.
In the first file, we have a list of nearly 15000 lines of post codes with the corresponding
city names plus additional information. Here are some arbitrary lines of this file:

68309,"Mannheim",8222,"Mannheim",8,"Baden-Wrttemberg"
68519,"Viernheim",6431,"Bergstraße",6,"Hessen"
68526,"Ladenburg",8226,"Rhein-Neckar-Kreis",8,"Baden-Württemberg"
68535,"Edingen-Neckarhausen",8226,"Rhein-Neckar-Kreis",8,"Baden-Württemberg"

The other file contains a list of the 19 largest German cities. Each line consists of the rank, the name of the city,
the population, and the state (Bundesland):

1.  Berlin                3.382.169 Berlin
2.  Hamburg         1.715.392 Hamburg
3.  München         1.210.223 Bayern
4.  Köln              962.884 Nordrhein-Westfalen
5.  Frankfurt am Main 646.550 Hessen
6.  Essen             595.243 Nordrhein-Westfalen
7.  Dortmund          588.994 Nordrhein-Westfalen
8.  Stuttgart         583.874 Baden-Württemberg
9.  Düsseldorf        569.364 Nordrhein-Westfalen
10. Bremen            539.403 Bremen
11. Hannover          515.001 Niedersachsen
12. Duisburg          514.915 Nordrhein-Westfalen
13. Leipzig           493.208 Sachsen
14. Nürnberg          488.400 Bayern
15. Dresden           477.807 Sachsen
16. Bochum            391.147 Nordrhein-Westfalen
17. Wuppertal         366.434 Nordrhein-Westfalen
18. Bielefeld         321.758 Nordrhein-Westfalen
19. Mannheim          306.729 Baden-Württemberg

Our task is to create a list with the top 19 cities, with the city names accompanied by the postal code.
If you want to test the following program, you have to save the list above in a file called largest_cities_germany.txt
and you have to download and save the list of German post codes
"""

import re

fh_post_codes = open("C:\Users\Wei\Desktop\post_codes_germany.txt")
PLZ = {}
for line in f:
    #print line
    (post_code, city, rest) = line.split(",", 2)
    PLZ[city.strip("\"")] = post_code

fh_largest_cities = open("C:\Users\Wei\Desktop\largest_cities_germany.txt")

for line in fh_largest_cities:
    re_obj = re.search(r"^[0-9]{1,2}\.\s+([\wÄÖÜäöüß\s]+\w)\s+[0-9]",line)
    city = re_obj.group(1)
    print city, PLZ[city]

# Another Comprehensive Example
"""
We want to present another real life example in our Python course. A regular expression for UK postcodes.
We write an expression, which is capable of recognizing the postal codes or postcodes of the UK.

Postcode units consist of between five and seven characters, which are separated into two parts by a space.
The two to four characters before the space represent the so-called outward code or out code intended to
direct mail from the sorting office to the delivery office. The part following the space, which consists of a digit
followed by two uppercase characters, comprises the so-called inward code, which is needed to sort mail at the
final delivery office. The last two uppercase characters can be only one of these ABDEFGHJLNPQRSTUWXYZ.

The outward code can have the form: One or two uppercase characters, followed by either a digit or the letter R,
optionally followed by an uppercase character or a digit. (We do not consider all the detailed led rules for
postcodes, i.e only certain character sets are valid depending on the position and the context.)
A regular expression for matching this superset of UK postcodes looks like this:
    r"\b[A-Z]{1,2}[0-9R][0-9A-Z]? [0-9][ABD-HJLNP-UW-Z]{2}\b"

"""

import re

example_codes = ["SW1A 0AA", # House of Commons
                                 "SW1A 1AA", # Buckingham Palace
                                 "SW1A 2AA", # Downing Street
                                 "BX3 2BB", # Barclays Bank
                                 "DH98 1BT", # British Telecom
                                 "N1 9GU", # Guardian Newspaper
                                 "E98 1TT", # The Times
                                 "TIM E22", # a fake postcode
                                 "A B1 A22", # not a valid postcode
                                 "EC2N 2DB", # Deutsche Bank
                                 "SE9 2UG", # University of Greenwhich
                                 "N1 0UY", # Islington, London
                                 "EC1V 8DS", # Clerkenwell, London
                                 "WC1X 9DT", # WC1X 9DT
                                 "B42 1LG", # Birmingham
                                 "B28 9AD", # Birmingham
                                 "W12 7RJ", # London, BBC News Centre
                                 "BBC 007" # a fake postcode
                                ]

pc_re = r"[A-z]{1,2}[0-9R][0-9A-Z]? [0-9][ABD-HJLNP-UW-Z]{2}" # [A-z]All Uppercase and lowercase a to z letters

for postcode in example_codes:
    r = re.search(pc_re, postcode)
    if r:
        print postcode + " matched!"
    else:
        print postcode + " is not a valid postcode!"


#http://www.endmemo.com/python/regex.php








