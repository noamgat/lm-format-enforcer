# regex engine in Python
# main program
# xiayun.sun@gmail.com
# 06-JUL-2013
# Supporting: alteration(|), concatenation, repetitions (* ? +), parentheses
#
# TODO: 
#     more rigorous bnf grammar for regex                 DONE
#     add . 
#     better unit tests                                   DONE
#     backreferences?                                     NO
#     convert to DFA
#     draw NFA in debug mode using Graphviz
#     return positions of match


from .parse import Lexer, Parser, Handler

def compile(p, debug = False):
    
    def print_tokens(tokens):
        for t in tokens:
            print(t)

    lexer = Lexer(p)
    parser = Parser(lexer)
    tokens = parser.parse()

    handler = Handler()
    
    if debug:
        print_tokens(tokens) 

    nfa_stack = []
    
    for t in tokens:
        handler.handlers[t.name](t, nfa_stack)
    
    assert len(nfa_stack) == 1
    return nfa_stack.pop() 

