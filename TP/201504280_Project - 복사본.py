# -*- coding: utf-8 -*-
from string import letters, digits, whitespace
import collections

class CuteType:
    INT = 1
    ID = 4

    MINUS = 2
    PLUS = 3

    L_PAREN = 5
    R_PAREN = 6

    TRUE = 8
    FALSE = 9

    TIMES = 10
    DIV = 11

    LT = 12
    GT = 13
    EQ = 14
    APOSTROPHE = 15

    DEFINE = 20
    LAMBDA = 21
    COND = 22
    QUOTE = 23
    NOT = 24
    CAR = 25
    CDR = 26
    CONS = 27
    ATOM_Q = 28
    NULL_Q = 29
    EQ_Q = 30

    KEYWORD_LIST = ('define', 'lambda', 'cond', 'quote', 'not', 'car', 'cdr', 'cons',
                    'atom?', 'null?', 'eq?')

    BINARYOP_LIST = (DIV, TIMES, MINUS, PLUS, LT, GT, EQ)
    BOOLEAN_LIST = (TRUE, FALSE)


def check_keyword(token):
    """
    :type token:str
    :param token:
    :return:
    """
    if token.lower() in CuteType.KEYWORD_LIST:
        return True
    return False


def _get_keyword_type(token):
    return {
        'define': CuteType.DEFINE,
        'lambda': CuteType.LAMBDA,
        'cond': CuteType.COND,
        'quote': CuteType.QUOTE,
        'not': CuteType.NOT,
        'car': CuteType.CAR,
        'cdr': CuteType.CDR,
        'cons': CuteType.CONS,
        'atom?': CuteType.ATOM_Q,
        'null?': CuteType.NULL_Q,
        'eq?': CuteType.EQ_Q
    }[token]


CUTETYPE_NAMES = dict((eval(attr, globals(), CuteType.__dict__), attr) for attr in dir(
    CuteType()) if not callable(attr) and not attr.startswith('__'))


class Token(object):
    def __init__(self, type, lexeme):
        """
        :type type:CuteType
        :type lexeme: str
        :param type:
        :param lexeme:
        :return:
        """
        if check_keyword(lexeme):
            self.type = _get_keyword_type(lexeme)
            self.lexeme = lexeme
        else:
            self.type = type
            self.lexeme = lexeme
        # print type

    def __str__(self):
        # return self.lexeme
        return '[' + CUTETYPE_NAMES[self.type] + ': ' + self.lexeme + ']'

    def __repr__(self):
        return str(self)


class Scanner:

    def __init__(self, source_string=None):
        """
        :type self.__source_string: str
        :param source_string:
        """
        self.__source_string = source_string
        self.__pos = 0
        self.__length = len(source_string)
        self.__token_list = []

    def __make_token(self, transition_matrix, build_token_func=None):
        old_state = 0
        self.__skip_whitespace()
        temp_char = ''
        return_token = ''
        while not self.eos():
            temp_char = self.get()
            if old_state == 0 and temp_char in (')', '('):
                return_token = temp_char
                old_state = transition_matrix[(old_state, temp_char)]
                break

            return_token += temp_char
            old_state = transition_matrix[(old_state, temp_char)]
            next_char = self.peek()
            if next_char in whitespace or next_char in ('(', ')'):
                break

        return build_token_func(old_state, return_token)

    def scan(self, transition_matrix, build_token_func):
        while not self.eos():
            self.__token_list.append(self.__make_token(
                transition_matrix, build_token_func))
        return self.__token_list

    def pos(self):
        return self.__pos

    def eos(self):
        return self.__pos >= self.__length

    def skip(self, pattern):
        while not self.eos():
            temp_char = self.peek()
            if temp_char in pattern:
                temp_char = self.get()
            else:
                break

    def __skip_whitespace(self):
        self.skip(whitespace)

    def peek(self, length=1):
        return self.__source_string[self.__pos: self.__pos + length]

    def get(self, length=1):
        return_get_string = self.peek(length)
        self.__pos += len(return_get_string)
        return return_get_string


class CuteScanner(object):

    transM = {}

    def __init__(self, source):
        """
        :type source:str
        :param source:
        :return:
        """
        self.source = source
        self._init_TM()

    def _init_TM(self):
        for alpha in letters:
            self.transM[(0, alpha)] = 4
            self.transM[(4, alpha)] = 4

        for digit in digits:
            self.transM[(0, digit)] = 1
            self.transM[(1, digit)] = 1
            self.transM[(2, digit)] = 1
            self.transM[(4, digit)] = 4

        self.transM[(0, '_')] = 1
        self.transM[(1, '_')] = 1
        self.transM[(2, '_')] = 1
        self.transM[(4, '_')] = 4

        self.transM[(4, '?')] = 16
        self.transM[(0, '-')] = 2
        self.transM[(0, '+')] = 3
        self.transM[(0, '(')] = 5
        self.transM[(0, ')')] = 6

        self.transM[(0, '#')] = 7
        self.transM[(7, 'T')] = 8
        self.transM[(7, 'F')] = 9

        self.transM[(0, '/')] = 11
        self.transM[(0, '*')] = 10

        self.transM[(0, '<')] = 12
        self.transM[(0, '>')] = 13
        self.transM[(0, '=')] = 14
        self.transM[(0, "'")] = 15

    def tokenize(self):

        def build_token(type, lexeme): return Token(type, lexeme)
        cute_scanner = Scanner(self.source)
        return cute_scanner.scan(self.transM, build_token)


class TokenType():
    INT = 1
    ID = 4
    MINUS = 2
    PLUS = 3
    LIST = 5
    TRUE = 8
    FALSE = 9
    TIMES = 10
    DIV = 11
    LT = 12
    GT = 13
    EQ = 14
    APOSTROPHE = 15
    DEFINE = 20
    LAMBDA = 21
    COND = 22
    QUOTE = 23
    NOT = 24
    CAR = 25
    CDR = 26
    CONS = 27
    ATOM_Q = 28
    NULL_Q = 29
    EQ_Q = 30

NODETYPE_NAMES = dict((eval(attr, globals(), TokenType.__dict__), attr) for attr in dir(
    TokenType()) if not callable(attr) and not attr.startswith('__'))

class Node (object):

    def __init__(self, type, value=None):
        self.next = None
        self.value = value
        self.type = type

    def set_last_next(self, next_node):
        if self.next is not None:
            self.next.set_last_next(next_node)

        else:
            self.next = next_node

    def __str__(self):
        result = ''

        if self.type is TokenType.ID:
            result = '[' + NODETYPE_NAMES[self.type] + ':' + self.value + ']'
        elif self.type is TokenType.INT:
            result = '['+NODETYPE_NAMES[self.type]+':' + self.value + ']'
        elif self.type is TokenType.LIST:
            if self.value is not None:
                if self.value.type is TokenType.QUOTE:
                    result = str(self.value)
                else:
                    result = '(' + str(self.value) + ')'
            else:
                result = '(' + str(self.value) + ')'
        elif self.type is TokenType.QUOTE:
            result = "\'"
        else:
            result = '['+NODETYPE_NAMES[self.type]+']'

        # fill out
        if self.next is not None:
            return result + ' ' + str(self.next)
        else:
            return result

    def __call__(self,arg=None):
        return Node(self.type,self.value)

class BasicPaser(object):

    def __init__(self, token_list):
        """
        :type token_list:list
        :param token_list:
        :return:
        """
        self.token_iter = iter(token_list)

    def _get_next_token(self):
        """
        :rtype: Token
        :return:
        """
        next_token = next(self.token_iter, None)
        if next_token is None:
            return None
        return next_token

    def parse_expr(self):
        """
        :rtype : Node
        :return:
        """
        token = self._get_next_token()

        '"":type :Token""'
        if token is None:
            return None
        result = self._create_node(token)
        return result

    def _create_node(self, token):
        if token is None:
            return None
        elif token.type is CuteType.INT:
            return Node(TokenType.INT,  token.lexeme)
        elif token.type is CuteType.ID:
            return Node(TokenType.ID,   token.lexeme)
        elif token.type is CuteType.L_PAREN:
            return Node(TokenType.LIST, self._parse_expr_list())
        elif token.type is CuteType.R_PAREN:
            return None
        elif token.type in CuteType.BOOLEAN_LIST:
            return Node(token.type)
        elif token.type in CuteType.BINARYOP_LIST:
            return Node(token.type, token.lexeme)
        elif token.type is CuteType.QUOTE:
            return Node(TokenType.QUOTE, token.lexeme)
        elif token.type is CuteType.APOSTROPHE:
            node = Node(TokenType.LIST, Node(TokenType.QUOTE, token.lexeme))
            node.value.next = self.parse_expr()
            return node
        elif check_keyword(token.lexeme):
            return Node(token.type, token.lexeme)

    def _parse_expr_list(self):
        head = self.parse_expr()
        '"":type :Node""'
        if head is not None:
            head.next = self._parse_expr_list()
        return head


def run_list(root_node):
    """
    :type root_node: Node
    """
    op_code_node = root_node.value
    return run_func(op_code_node)(root_node)


def run_func(op_code_node):
    """
    :type op_code_node:Node/
    """
    def quote(node):
        return node

    def strip_quote(node):
        """
        :type node: Node
        """
        if node.type is TokenType.LIST:
            if node.value is TokenType.QUOTE or TokenType.APOSTROPHE:
                return node.value.next
        if node.type is TokenType.QUOTE:
            return node.next
        return node

    def cons(node):
        """
        :type node: Node
        """
        l_node = node.value.next
        r_node = l_node.next
        r_node = run_expr(r_node)
        l_node = run_expr(l_node)
        new_r_node = r_node
        new_l_node = l_node
        new_r_node = strip_quote(new_r_node)
        new_l_node = strip_quote(new_l_node)
        new_l_node.next = new_r_node.value

        return create_new_quote_list(new_l_node, True)

    def car(node):
        l_node = run_expr(node.value.next)
        if l_node.type is TokenType.ID :
            l_node = var_get(l_node.value)
        result = strip_quote(l_node).value
        if result.type is not TokenType.LIST:
            return result
        return create_new_quote_list(result)

    def cdr(node):
        """
        :type node: Node
        """
        l_node = node.value.next
        if l_node.type is TokenType.ID :
            l_node = var_get(l_node.value)
        l_node = run_expr(l_node)
        if l_node.type is TokenType.LIST and l_node.value.type is TokenType.LIST :
            l_node = l_node.value
        new_r_node = strip_quote(l_node)
        return create_new_quote_list(new_r_node.value.next, True)

    def null_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node).value
        if new_l_node is None:
            return Node(TokenType.TRUE)
        else:
            return Node(TokenType.FALSE)

    def atom_q(node):
        l_node = run_expr(node.value.next)
        new_l_node = strip_quote(l_node)

        if new_l_node.type is TokenType.LIST:
            if new_l_node.value is None:
                return Node(TokenType.TRUE)
            return Node(TokenType.FALSE)
        else:
            return Node(TokenType.TRUE)

    def eq_q(node):
        l_node = node.value.next
        r_node = l_node.next
        new_l_node = strip_quote(run_expr(l_node))
        new_r_node = strip_quote(run_expr(r_node))

        if (new_l_node.type or new_r_node.type) is not TokenType.INT:
            return Node(TokenType.FALSE)
        if new_l_node.value == new_r_node.value:
            return Node(TokenType.TRUE)
        return Node(TokenType.FALSE)

    # Fill Out
    # table을 보고 함수를 작성하시오

    #run_binary
    def run_binary(op_node):
        left = op_node.value.next  # op_node.value == operation node
        right = left.next
        left = run_expr(left)
        right = run_expr(right)
        #변수 있으면 varTable에서 가져오는 부분
        if left.type is TokenType.ID :
            left = Node(TokenType.INT,var_get(left.value))
        if right.type is TokenType.ID :
            right = Node(TokenType.INT, var_get(right.value))
        #연산 수행
        if op_node.value.type is TokenType.MINUS:
            op_result = int(left.value) - int(right.value)
            result_node = Node(TokenType.INT, op_result)
        elif op_node.value.type is TokenType.PLUS:
            op_result = int(left.value) + int(right.value)
            result_node = Node(TokenType.INT, op_result)
        elif op_node.value.type is TokenType.TIMES:
            op_result = int(left.value) * int(right.value)
            result_node = Node(TokenType.INT, op_result)
        elif op_node.value.type is TokenType.DIV:
            op_result = int(left.value) / int(right.value)
            result_node = Node(TokenType.INT, op_result)
        elif op_node.value.type is TokenType.LT:
            if int(left.value) < int(right.value):
                result_node = Node(TokenType.TRUE, True)
            else:
                result_node = Node(TokenType.FALSE, False)
        elif op_node.value.type is TokenType.GT:
            if int(left.value) > int(right.value):
                result_node = Node(TokenType.TRUE, True)
            else:
                result_node = Node(TokenType.FALSE, False)
        elif op_node.value.type is TokenType.EQ:
            if int(left.value) == int(right.value):
                result_node = Node(TokenType.TRUE, True)
            else:
                result_node = Node(TokenType.FALSE, False)
        return result_node

    #산술연산
    def minus(node):
        return run_binary(node)

    def plus(node):
        return run_binary(node)

    def multiple(node):
        return run_binary(node)

    def divide(node):
        return run_binary(node)

    #관계연산
    def lt(node) :
        return run_binary(node)

    def gt(node) :
        return run_binary(node)

    def eq(node) :
        return run_binary(node)

    def not_op(node) :
        condition = run_expr(node.value.next)
        if condition.type is TokenType.TRUE :
            return Node(TokenType.FALSE,False)
        elif condition.type is TokenType.FALSE :
            return Node(TokenType.TRUE,True)
        else :
            print("not_operation error!")

    def cond(node):
        l_node = node.value.next

        if l_node is not None:
            result = run_cond(l_node)
            return result
        else:
            print('cond null error!')

    def run_cond(node):
        """
        :type node: Node
        """
        #Fill Out
        condition = node.value
        result = condition.next
        exam_condition = run_expr(condition)
        if node.value.type is TokenType.QUOTE :
            strip_quote(node)

        if exam_condition.type is TokenType.TRUE :
            return result
        elif exam_condition.type is TokenType.FALSE :
            if condition.next is not None:
                return run_cond(node.next)
            else :
                return None

    def make_func_table(node):
        var_val = node
        function_table = collections.OrderedDict()  # 함수 테이블을 생성한다.
        '''위에 코드는 expr, 변수1, 변수2.. 순서로 저장하기 위해 일반 딕셔너리가 아닌, collections모듈의 ordered dictionary를 사용했다.'''
        function_table['expr'] = var_val  # expr은 함수 내용을 저장한다. 즉, expression을 저장한다.
        param = var_val.value.next
        param = param.value
        i = 0
        while (param is not None):
            function_table[param.value] = None  # 파라미터들을 저장한다. 키값은 파라미터 이름이고, value값은 None으로 저장된다.
            i = i + 1
            param = param.next
        return function_table

    def var_def(node):
        #변수의 이름을 var_name, 변수의 값을 var_val에 저장
        var_name = node.value.next
        var_val = var_name.next
        '''
        만약 var_val의 타입이 List이면 run_expr을 한다(define a (- 1 2))같은 것들을 처리하기 위함)
        단, 변수에 저장할 값이 LAMBDA인 경우에는 run_expr하지 않는다. (define plus (lambda (x) (+ x 1)))처럼 expression을 그대로 저장해야 하기 때문에 run_expr로 expression을 처리해 버리면 안된다.
        '''
        if(var_val.type is TokenType.LIST and var_val.value.type is not TokenType.LAMBDA):
            var_val = run_expr(var_val)
        if var_val.type is TokenType.ID : #만약 value가 문자면 변수인 것이므로, 해당하는 변수값을 찾아서 value로 해준다. (define b 10) (define test b) --> test == 10 이런거 처리하기 위함임.
            var_val = var_get(var_val.value)
        var_name = var_name.value
        if var_val is None :
            var_val = None
        elif type(var_val) is int :
            var_val = Node(TokenType.INT,var_val)
        elif type(var_val) is str :
            var_val = Node(TokenType.ID,var_val)
        elif (var_val.type is TokenType.INT) or (var_val.type is TokenType.ID) :
            var_val = var_val
        elif var_val.type is TokenType.QUOTE :
            var_val = Node(TokenType.LIST,var_val)
        elif var_val.value.type is TokenType.QUOTE :
            var_val = var_val  #상수, quote리스트를 저장하는 경우는 여기까지
        else : #람다인 경우
            var_val = var_val.value

        if var_name is "null": print "변수명을 null로 선언할 수 없습니다."
        '''
        아래부터는 함수를 저장할 때 추가적으로 수행해주는 부분이다.
        '''
        if (type(var_val) is not str) and (type(var_val) is not int) and (var_val is not None) :
            if(var_val.type is TokenType.LAMBDA) :
                var_val = Node(TokenType.LIST,var_val) #LIST노드로 감싸준다. 현재 lamda (x) (+ x 1)형태이기 때문에 (lamda (x) (+ x 1))형태, 즉 expression형태로 만들기 위함이다
                var_val = make_func_table(var_val)

        if(varTable.get(var_name)!=None): #만약 val_name이 이미 변수테이블에 존재한다면(중복되는 이름이 있다면)
            del varTable[var_name] #기존의 변수 내용을 삭제한다

        varTable[var_name]=var_val #변수 테이블에 var_name, var_val 쌍을 저장한다
        if type(varTable[var_name]) is collections.OrderedDict : #함수이면 nasted여부를 확인
            func_table = varTable[var_name]
            expr = func_table['expr']
            is_first_list = True
            while expr is not None :
                if expr.value.type is TokenType.DEFINE :
                    nasted_func_name = expr.value.next.value
                    varTable[var_name]['nasted'] = nasted_func_name
                    break
                if is_first_list is True :
                    expr = expr.value.next
                    is_first_list = False
                else :
                    expr = expr.next
            return
    def run_user_func(func_name,node) :
        #에러검사
        if(var_get(func_name) is None):
            if type(func_name) is str :
                print func_name + " is an undefined function."
            else :
                print func_name.value + " is an undefined function."
            return Node(TokenType.FALSE,False)
        if type(var_get(func_name)) is collections.OrderedDict :
            func_table = var_get(func_name)
        else:
            func_table = varTable[func_name] #func_table:함수테이블
        func_expr = func_table.get('expr') #func_expr:함수 내용 ex. (lambda (x y) (+ x y))
        param = node
        # 재귀호출과 함수인자를 처리하기 위한 부분. 이 부분이 있어야 인자가 함수거나, 연산일때 적절히 처리할 수 있다.
        # ex)인자가 (cdr ls)일때 run_expr해서 적절한 인자인 ID노드 또는 INT노드를 반환해줌
        # or안쓰고 조건문 중첩해서 쓴 이유는 단축계산 때문이다.
        if node.type is not TokenType.INT:
            if node.type is not TokenType.ID :
                if node.value.type is not TokenType.QUOTE :
                    param = run_expr(node)

        run_target_expr = Node(func_expr.type,func_expr.value)
        # nasted함수 처리 부분
        if 'nasted' in func_table:  # nasted필드가 있으면 중첩함수다.
            prev_position = None
            target_position = run_target_expr
            next_position = run_target_expr.value.next
            while next_position is not None :
                if target_position.value.type is TokenType.DEFINE : #define문장을 만나면 run_expr한다(define문장을 실행시켜 내부 함수를 정의해준다)
                    target_position.next = None
                    run_expr(target_position)
                    break
                else : #prev -> target -> next형태로 이동한다.
                    prev_position = target_position
                    target_position = next_position
                    next_position = next_position.next
            prev_position.next = next_position

        run_target_expr.next = param #인자 붙임 ex. (lambda (x y) (+ x y)) 1 2
        run_target_expr = Node(TokenType.LIST,run_target_expr) #리스트로 포장해 lambda식 형태로 만듬 ex. ((lambda (x y) (+x y)) 1 2)

        #함수 실행 후 결과값 반환(함수는 lam에 의해 처리됨)
        if type(var_get(func_name)) is collections.OrderedDict :
            if varTable.get(func_name) is None :
                result = lam(run_target_expr)
            else :
                result= lam(run_target_expr,func_name)
        else:
            result = lam(run_target_expr, func_name)

        #nasted함수 실행 후 nasted함수 제거하는 과정
        if varTable.get(func_name) is not None:
            if 'nasted' in varTable.get(func_name):
                nasted_func_name = varTable.get(func_name).get('nasted')
                del varTable[nasted_func_name]

        return result


    def lam(node, func_name='null'):
        func_expr = node.value.value
        func_expr = Node(TokenType.LIST, func_expr)
        if func_name is 'null' :
            global null_func_numb
            func_name = 'null'+str(null_func_numb)
            varTable[func_name]= make_func_table(func_expr)
            null_func_numb += 1
        func_table = varTable[func_name]

        # 매개변수 바인딩
        argument = []
        target = node.value.next
        while True:
            if target.type is TokenType.ID : #인자로 변수가 전달된 경우를 처리함. ex) 15번 test case
                target.value = var_get(target.value)
                target.type = TokenType.INT
            target_node = Node(target.type,target.value)
            # 함수 인자를 사용하기 위한 부분(변수를 varTable에서 찾았는데 결과값이 함수테이블로 전달된 경우)
            if type(target_node.value) is not str :
                if type(target_node.value) is collections.OrderedDict :
                    target_node = target_node.value
            argument.append(target_node)  # argument 리스트에 전달된 인자들을 하나씩 저장
            target = target.next
            if target is None:
                break
        argument.reverse()  # 리스트의 pop연산이 리스트 뒤에서부터 뽑기 때문에 리스트를 reverse해준다.
        try:
            for param_name in func_table.keys():
                if param_name is 'expr' or param_name is 'nasted': continue  # 키값 expr이거나, nasted일때는 매개변수 아니므로 continue
                func_table[param_name] = argument.pop()  # 리스트에서 인자를 하나씩 꺼내 함수테이블의 파라미터에 바인딩한다.
        except:
            print func_name + "함수의 파라미터 개수와 인자의 개수가 일치하지 않습니다."
            return None

        func_expr = func_expr.value.next.next
        funcStack.append(func_name) #함수 이름을 funcStack에 push
        result = run_expr(func_expr) #함수 실행
        #재귀호출을 위한 부분
        if result.type is not TokenType.INT :
            if varTable.get(result.value.value) is not None : #만약 함수 실행 결과중에 varTable에 기록된 함수가 있고
                recursion_func_name = result.value.value
                if type(varTable.get(recursion_func_name)) is collections.OrderedDict : #그것이 함수라면 재귀호출을 수행할 시점이다.
                    result = run_user_func(recursion_func_name,result.value.next)
        result = run_expr(result)
        funcStack.pop() #함수 이름을 funcStack에서 pop
        #임시로 생성한 함수인 경우 함수 자체를 삭제, 사용자 정의 함수인 경우 파라미터 초기화
        if func_name is 'null' :
            del varTable['null']
        else :
            for param_name in func_table.keys():
                if param_name is 'expr' or param_name is 'nasted':continue
                func_table[param_name] = None
        return result #결과값 반환

    def create_new_quote_list(value_node, list_flag=False):
        """
        :type value_node: Node
        """
        quote_list = Node(TokenType.QUOTE, 'quote')
        wrapper_new_list = Node(TokenType.LIST, quote_list)
        if value_node is None:
            pass
        elif value_node.type is TokenType.LIST:
            if list_flag:
                inner_l_node = Node(TokenType.LIST, value_node)
                quote_list.next = inner_l_node
            else:
                quote_list.next = value_node
            return wrapper_new_list
        new_value_list = Node(TokenType.LIST, value_node)
        quote_list.next = new_value_list
        return wrapper_new_list

    table = {}
    table['cons'] = cons
    table["'"] = quote
    table['quote'] = quote
    table['cdr'] = cdr
    table['car'] = car
    table['eq?'] = eq_q
    table['null?'] = null_q
    table['atom?'] = atom_q
    table['not'] = not_op
    table['+'] = plus
    table['-'] = minus
    table['*'] = multiple
    table['/'] = divide
    table['<'] = lt
    table['>'] = gt
    table['='] = eq
    table['cond'] = cond
    table['define'] = var_def
    table['lambda'] = lam

    if op_code_node.type is TokenType.LIST :
        op_code_node = op_code_node.value

    if(table.get(op_code_node.value) is None) :
        return run_user_func(op_code_node.value,op_code_node.next)
    else:
       return table[op_code_node.value]



def run_expr(root_node):
    """
    :type root_node : Node
    """
    if root_node is None:
        return None

    if root_node.type is TokenType.ID:
        return root_node
    elif root_node.type is TokenType.INT:
        return root_node
    elif root_node.type is TokenType.TRUE:
        return root_node
    elif root_node.type is TokenType.FALSE:
        return root_node
    elif root_node.type is TokenType.LIST:
        return run_list(root_node)
    else:
        print 'Run Expr Error'
    return None


def print_node(node):
    """
    "Evaluation 후 결과를 출력하기 위한 함수"
    "입력은 List Node 또는 atom"
    :type node: Node
    """
    def print_list(node):
        """
        "List노드의 value에 대해서 출력"
        "( 2 3 )이 입력이면 2와 3에 대해서 모두 출력함"
        :type node: Node
        """
        def print_list_val(node):
            if node.next is not None:
                return print_node(node)+' '+print_list_val(node.next)
            return print_node(node)

        if node.type is TokenType.LIST:
            if node.value is None:
                return '( )'
            if node.value.type is TokenType.QUOTE:
                return print_node(node.value)
            return '('+print_list_val(node.value)+')'

    if node is None:
        return ''
    if node.type in [TokenType.ID, TokenType.INT]:
        return node.value
    if node.type is TokenType.TRUE:
        return '#T'
    if node.type is TokenType.FALSE:
        return '#F'
    if node.type is TokenType.PLUS:
        return '+'
    if node.type is TokenType.MINUS:
        return '-'
    if node.type is TokenType.TIMES:
        return '*'
    if node.type is TokenType.DIV:
        return '/'
    if node.type is TokenType.GT:
        return '>'
    if node.type is TokenType.LT:
        return '<'
    if node.type is TokenType.EQ:
        return '='
    if node.type is TokenType.LIST:
        return print_list(node)
    if node.type is TokenType.ATOM_Q:
        return 'atom?'
    if node.type is TokenType.CAR:
        return 'car'
    if node.type is TokenType.CDR:
        return 'cdr'
    if node.type is TokenType.COND:
        return 'cond'
    if node.type is TokenType.CONS:
        return 'cons'
    if node.type is TokenType.LAMBDA:
        return 'lambda'
    if node.type is TokenType.NULL_Q:
        return 'null?'
    if node.type is TokenType.EQ_Q:
        return 'eq?'
    if node.type is TokenType.NOT:
        return 'not'
    if node.type is TokenType.QUOTE:
        return "'"+print_node(node.next)

def Print_method(input):
    test_cute = CuteScanner(input)
    test_tokens = test_cute.tokenize()
    test_basic_paser = BasicPaser(test_tokens)
    node = test_basic_paser.parse_expr()
    cute_inter = run_expr(node)

    if type(cute_inter) is Node and cute_inter.type is TokenType.ID :
        if varTable.get(cute_inter.value) is not None :
            cute_inter.value = varTable.get(cute_inter.value)
            if type(cute_inter) is Node :
                cute_inter = Node(cute_inter.value.type,cute_inter.value.value)

    if cute_inter is None :
        return
    else :
        print "..."+str(print_node(cute_inter))

def var_get(var_name):
    if funcStack.__len__() > 0 : #funcStack이 있을 경우(함수 실행중인 경우)
        funcIdx = funcStack.__len__()-1 #인덱스는 funcStack의 맨 마지막(현재 실행중인 함수)로 초기화됨
        '''
        먼저 자신의 영역에서 변수를 찾는다. 만약 변수를 찾지 못했다면,
        static parent의 영역에서 변수를 찾는다.
        '''
        while True :
            if funcIdx < 0:  # 전역을 제외한 모든 static parent를 조사했으면 break
                break
            var_value = varTable[funcStack[funcIdx]].get(var_name)
            if var_value is not None :
                if type(var_value) is collections.OrderedDict : #변수를 찾았으면 break(변수가 함수테이블을 저장하고 있는 경우임. 이 코드 안넣어주면 ordered dictionary는 type이 없어서 아래 if문에서 에러남.)
                    break
                if var_value.type is TokenType.ID : #변수를 찾긴 했는데 여전히 변수이면
                    funcIdx = funcIdx-1 #static parent를 다시 조사
                    continue
                break #변수를 찾았으면 break
            funcIdx = funcIdx-1 #변수를 못찾았으면 static parent를 조사
        if(var_value is None) :
            var_value = varTable.get(var_name) #전역에서 변수 찾음
            if var_value is None: #그래도 없으면 None반환
                return None
    else :
        var_value = varTable.get(var_name) #함수 실행중이지 않은 경우 그냥 전역에서 찾는다.
    try :
        if type(var_value) is collections.OrderedDict :
            return var_value #찾은 변수값이 함수인 경우 함수테이블을 반환
        elif (type(var_value) is str) or (type(var_value) is int) :
            return var_value #찾은 변수값 반환
        elif (var_value.type is TokenType.LIST) and (var_value.value.type is TokenType.QUOTE) :
            return var_value #quote list는 quote list반환
        else :
            return var_value.value
    except :
        print "There was an error getting the value of the variable"
        return None
def run_console():
    while True :
        try :
            user_expr =  raw_input(">")
            Print_method(user_expr)
            if user_expr == "exit" : break
            elif user_expr == "t1" :
                Print_method('(define a 1)')
            elif user_expr == "t2" :
                Print_method("(define b '(1 2 3)")
            elif user_expr == "t3" :
                Print_method("(define c (- 5 2)")
            elif user_expr == "t4" :
                Print_method("(define d '(+ 2 3)")
            elif user_expr == "t5" :
                Print_method("(define test b)")
            elif user_expr == "t6" :
                Print_method("(+ a 3)")
            elif user_expr == "t7" :
                Print_method("(define a 2)")
                Print_method("(* a 4)")
            elif user_expr == "t8" :
                Print_method("((lambda (x) (* x -2)) 3)")
            elif user_expr == "t9" :
                Print_method("((lambda (x) (/ x 2)) a)")
            elif user_expr == "t10" :
                Print_method("((lambda (x y) (* x y)) 3 5)")
            elif user_expr == "t11" :
                Print_method("((lambda (x y) (* x y)) a 5)")
            elif user_expr == "t12" :
                Print_method("(define plus1 (lambda (x) (+ x 1)))")
                Print_method("(plus1 3)")
            elif user_expr == "t13" :
                Print_method("(define mul1 (lambda (x) (* x a)))")
                Print_method("(mul1 a)")
            elif user_expr == "t14" :
                Print_method("(define plus2 (lambda (x) (+ (plus1 x) 1)))")
                Print_method("(plus2 4)")
            elif user_expr == "t15" :
                Print_method("(define plus3 (lambda (x) (+ (plus1 x) a)))")
                Print_method("(plus3 a)")
            elif user_expr == "t16" :
                Print_method("(define mul2 (lambda (x) (* (plus1 x) -2)))")
                Print_method("(mul2 7)")
            elif user_expr == "t17" :
                Print_method("(define lastitem (lambda (ls) (cond ((null? (cdr ls)) (car ls)) (#T (lastitem (cdr ls))))))")
                Print_method("(lastitem '(1 2 5))")
            elif user_expr == "t18":
                Print_method("(define square (lambda (x) (* x x)))")
                Print_method("(define yourfunc (lambda (x func) (func x)))")
                Print_method("(yourfunc 3 square)")
            elif user_expr == "t19":
                Print_method("(define square (lambda (x) (* x x)))")
                Print_method("(define mul_two (lambda (x) (* 2 x)))")
                Print_method("(define new_fun (lambda (fun1 fun2 x) (fun2 (fun1 x))))")
                Print_method("(new_fun square mul_two 10)")
            elif user_expr == "cubetest":
                Print_method(" (define cube (lambda (n) (define sqrt (lambda (n) (* n n ))) (* (sqrt n) n )))")
                Print_method("(cube 4)")
            elif user_expr == "t20" :
                Print_method(" (define cube (lambda (n) (define sqrt (lambda (n) (* n n ))) (* (sqrt n) n )))")
                Print_method("(sqrt 4)")
        except :
            print "Invalid input. Please re-enter."


varTable = dict() #변수를 저장할 테이블
funcStack = [] #함수 스택으로 사용
null_func_numb = 0

run_console()
#Test_All()
