import sys
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        import annotationlib
        print(namespace['__annotate_func__'](annotationlib.Format.VALUE))
        return super().__new__(mcs, name, bases, namespace)

class Test(metaclass=Meta):
    a: int
    b: 'str' = "test"
