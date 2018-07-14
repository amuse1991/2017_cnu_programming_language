
import random

class RecursionLinkedList(object):

    def __init__(self):
        self.head = None

    def _link_first(self, element):
        # connect newly created node the beginning of the list
        if self is not None:
            self.head = Node(element, self.head)
        else:
            self.head = Node(element, None)

    def _link_last(self, element, node):
        """
        :type node: Node
        :type element: char
        """
        # assignment(1) connect given node the next of the last linked node
        if node.next is None:  # create next node
            node.next = Node(element,None)
            if self.head == None :
                self.head = node.next
        else:  # visit next node
            return self._link_last(element,node.next)


    def _link_next(self, index, element):
        """
        :type index: Int
        :type element: Char
        :param index
        :param element
        """
        next = self.get_node(index).next
        self.get_node(index).next = Node(element, next)

    def _unlink_first(self):
        # unlink first node of list
        x = self.head
        """:type : Node"""
        element = x.element
        self.head = x.next
        return element

    def _unlink_next(self, pred):
        """
        :type pred: Node
        :param pred:
        """
        x = pred.next
        element = x.element
        pred.next = x.next
        return element

    def _get_node(self, index, x):
        """
        :type index:int
        :type x:Node
        :param index:
        :param x:
        """
        # assignment(2) Get nth(index) node
        if index == 0:  # return current node
            return x

        elif index > 0:  # return result of call _get_node
            return self._get_node(index-1,x.next)



    def get_node(self, index):
        return self._get_node(index, self.head)

    def __len__(self):
        if self.head is None:
            return 0
        return len(self.head)

    def add(self, element, index=None):
        if index is None:
            if self.head is None:
                self._link_first(element)
            else:
                self._link_last(element, self.head)
            return

        if index < 0 or index > len(self):
            print "ERROR"
        elif index == 0:
            self._link_first(element)
        else:
            self._link_next(index-1, element)

    def remove(self, index):
        if index < 0 or index > len(self):
            print "ERROR"
        elif index == 0:
            return self._unlink_first()
        else:
            return self._unlink_next(self._get_node(index - 1, self.head))

    def __str__(self):
        if self.head is None:
            return "List is null"
        return str(self.head)

    def _reverse(self, x, pred):
        """
        :type x: Node
        :type pred: Node
        :param x:
        """
        # assignment(5)
        # Fill out, Use recursion

        if x == None :
            self.head = pred
            return
        else :
            next = x.next
            x.next = pred
            pred = x
            x=next
            return _reverse(next,x)



    def reverse(self):
        self._reverse(self.head, None)

    def iter_selection_sort(self):
        current_node = self.head
        compare_node = self.head
        min_node = self.head

        while current_node.next is not None:
            while compare_node is not None:
                if min_node.element > compare_node.element:
                    min_node = compare_node
                compare_node = compare_node.next
            current_node.element, min_node.element = min_node.element, current_node.element
            current_node = current_node.next
            compare_node = current_node
            min_node = current_node

    def selection_sort(self):
        self._selection(self.head)

class Node(object):
    """
    :type element:char
    :type next: Node
    """

    def __init__(self, element, next):
        """
        :type element : char
        :type next : Node
        """
        self.element = element
        self.next = next

    def __str__(self):
        # assignment(3)
        if self.next is None:  # Return string of self.element
            return self.element

        else:  # Return self.element and string of next
            return self.element, str(self.next)




    def __len__(self):
        # assignment(4) Return size of entire node
        if self.next == None :
            return 0
        else :
            return(1+len(self))


def test_recursion_linked_list():
    INPUT = ['a', 'b', 'c', 'd', 'e']
    test_RLL = RecursionLinkedList()
    for i in INPUT:
        test_RLL.add(i)
    print str(test_RLL)
    print "List size is " + str(len(test_RLL))
    test_RLL.add('z', 0)
    print "List size is " + str(len(test_RLL))
    print str(test_RLL)
    print test_RLL.get_node(4).element
    print test_RLL.remove(0)
    print str(test_RLL)
    test_RLL.reverse()
    print str(test_RLL)



test_recursion_linked_list()

