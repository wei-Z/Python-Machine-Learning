# Python Regular Expressions
# In Python a regular expression search is typically written as:
#             match = re.search(pattern, string)

"""
The 'r' at the start of the pattern string designates a python "raw" string 
which passes through backslashes without change which is very handy 
for regular expressions (Java needs this feature badly!). 
I recommend that you always write pattern strings with the 'r' just as a habit.
"""

import re
str = 'an example word:cat!!'
match = re.search(r'word:\w\w\w', str)
# If- statement after search() tests if it succeeded
if match:
    print 'found', match.group() ## 'match.group() is the matching text (e.g. 'word:cat')
else:
    print 'did not find'
    
# Basic Patterns Examples
"""
The basic rulles of regular expression search for a pattern within a string are:
. The search proceeds through the string from start to end, stopping at the first match found.
. All of the pattern must be matched, but not all of the string.
. If match = re.search(pat, str) is successful, match is not None and in particular match.group() is the matching text.
"""

## Search for pattern 'iii' in string 'piiig'.
## All of the pattern must match, but it may appear anywhere.
## On success, match.group() is matched text.
match = re.search(r'iii', 'piiig') 
print match.group()
match = re.search(r'igs', 'piiig')
print match.group()

## . = any char but \n
match = re.search(r'..g', 'piiig') 
print match.group()

## \d = digit char, \w = word char
match = re.search(r'\d\d\d', 'p123g')
print match.group()
match = re.search(r'\w\w\w', '@@abcd!!')
print match.group()

# Repetition Examples

#Leftmost & Largest:
#First the search finds the leftmost match for the pattern, and second it tries to use up as much of the string 
#as possible -- i.e. + and * go as far as possible (the + and * are said to be "greedy").

## i+ = one or more i's, as many as possible.
match = re.search(r'pi+', 'piiig')
print match.group()

## Finds the first/leftmost solution, and within it drives the +
## as far as possible (aka 'leftmost and largest')
## In this example, note that it does not get to the second set of i's
match = re.search(r'i+', 'piigiiii') 
print match.group()

## \s* = zero or more whitespace chars
## Here look for 3 digits, possibley separated by whitespace.
match = re.search(r'\d\s*\d\s*\d', 'xx1 2    3xx')
print match.group()
match = re.search(r'\d\s*\d\s*\d', 'xx12   3xx')
print match.group()
match = re.search(r'\d\s*\d\s*\d', 'xx123xx')
print match.group()

## ^ = matches the start of string, so this fails:
match = re.search(r'^b\w+', 'foobar') 
print match.group()
## but without the ^ it succeeds:
match = re.search(r'b\w+', 'foobar')
print match.group()


# Emails Example
# Suppose you want to find the mail address inside the string 'xyz alice-b@google.com purple monkey'.
# We'll use this as a running example to demonstrate more regular expression features. Here's an attempt
# using the pattern r'\w+@\w+':

str = 'purple alice-b@google.com monkey dishwasher'
match = re.search(r'\w+@\w+', str)
if match:
    print match.group() ## b@google
#The search does not get the whole email address in this case because 
#the \w does not match the '-' or '.' in the address.     

# Square Brackets
"""
Square brackets can be used to indicate a set of chars, so [abc] matches 'a' or 'b' or 'c'. 
The codes \w, \s etc. work inside square brackets too with the one exception that dot (.) just means a literal dot. 
For the emails problem, the square brackets are an easy way to add '.' and '-' to the set of chars 
which can appear around the @ with the pattern r'[\w.-]+@[\w.-]+' to get the whole email address:
    
"""
match = re.search(r'[\w.-]+@[\w.-]+', str)
if match:
    print match.group() ## 'alice-b@google.com'

"""
(More square-bracket features) You can also use a dash to indicate a range, so [a-z] matches all lowercase 
letters. To use a dash without indicating a range, put the dash last, e.g. [abc-]. 
An up-hat (^) at the start of a square-bracket set inverts it, so [^ab] means any char except 'a' or 'b'.

"""

# Group Extraction

"""
The "group" feature of a regular expression allows you to pick out parts of the matching text. 
Suppose for the emails problem that we want to extract the username and host separately. 
To do this, add parenthesis ( ) around the username and host in the pattern, like this: r'([\w.-]+)@([\w.-]+)'. 
In this case, the parenthesis do not change what the pattern will match, instead they establish logical "groups" 
inside of the match text. On a successful search, match.group(1) is the match text corresponding to the 1st left
parenthesis, and match.group(2) is the text corresponding to the 2nd left parenthesis. The plain match.group() 
is still the whole match text as usual.
"""
str = 'purple alice-b@google.com monkey dishwasher'
match = re.search('([\w.-]+)@([\w.-]+)', str)
if match:
    print match.group()       ## 'alice-b@google.com' (the whole match)
    print match.group(1)     ## 'alice-b' (the username, group 1)
    print match.group(2)     ## 'google.com' (the host, group 2) 
# A common workflow with regular expressions is that you write a pattern for the thing you are looking for, 
# adding parenthesis groups to extract the parts you want.

# findall
"""

findall() is probably the single most powerful function in the re module. 
Above we used re.search() to find the first match for a pattern. findall() finds *all* the matches 
and returns them as a list of strings, with each string representing one match.

"""
## Suppose we have a text with many email addresses
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'

## Here re.findall() returns a list of all the found email strings
emails = re.findall(r'[\w\.-]+@[\w\.-]+', str) ## ['alice@google.com', 'bob@abc.com']
for email in emails:
    # do something with each found email string
    print email
    
# findall with Files
"""
For files, you may be in the habit of writing a loop to iterate over the lines of the file, and you could then 
call findall() on each line. Instead, let findall() do the iteration for you -- much better! 
Just feed the whole file text into findall() and let it return a list of all the matches in a single step (recall that 
f.read() returns the whole text of a file in a single string):

"""
# Open file
f = open('test.txt', 'r')
# Feed the file text into findall(); it returns a list of all the found strings
string = re.findall(r'some pattern', f.read())


#findall and Group
""" 

The parenthesis ( ) group mechanism can be combined with findall(). 
If the pattern includes 2 or more parenthesis groups, then instead of returning a list of strings, 
findall() returns a list of *tuples*. Each tuple represents one match of the pattern, and inside the tuple 
is the group(1), group(2) .. data. So if 2 parenthesis groups are added to the email pattern, 
then findall() returns a list of tuples, each length 2 containing the username and host, e.g. ('alice', 'google.com').

"""
import re
str = 'purple alice@google.com,  blah monkey bob@abc.com blah dishwasher'
tuples = re.findall(r'([\w\.-]+)@([\w\.-]+)', str)
print tuples ## [('alice', 'google.com'), ('bob', 'abc.com')]
for tuple in tuples:
    print tuple[0]  ## username
    print tuple[1]  ## host

# RE Workflow and Debug
# Options

"""
The re functions take options to modify the behavior of the pattern match.
The option flag is added as an extra argument to the search() or findall() etc.,
e.g. re.search(pat, str, re.IGNORECASE).

. IGNORECASE -- ignore upper/lowercase differences for matching, so 'a' matches both 'a' and 'A'.
. DOTALL -- allow dot (.) to match newline -- normally it matches anything but newline.
  This can trip you up -- you think .* matches everything, but by default it does not go past the end of
  a line. Note that \s (whitespace) includes newlines, so if you want to match a run of whitespace that
  may include a newline, you can just use \s*
. MULTILINE -- Within a string made of many lines, allow ^ and $ to match the start and end of each
   line. Normally ^/$ would just match the start and end of the whole string.

"""

# Greedy vs. Non-Greedy (optional)

# Substitution (optional)

"""
The re.sub(pat, replacement, str) function searches for all the instances of pattern in the given string,
and replaces them. The replacement string can include '\1', '\2' which refer to the text from group(1),
group(2), and so on from the original matching text.

Here's an example which searches for all the email addresses, and changes them to keep the user
(\1) but have yo-yo-dyne.com as the host.

"""
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
## re.sub(pat, replacement, str) -- returns new string with all replacements,
## \1 is group(1), \2 group(2) in the replacement
print re.sub(r'([\w\.-]+)@([\w\.-]+)', r'\1@yo-yo-dyne.com', str)
# # purple alice@yo-yo-dyne.com, blah monkey bob@yo-yo-dyne.com blah dishwasher

# Exercise
# Baby Names Exercise




