class Ban:
    def __init__(self, no=None):
        self.no = no
        self.student_list = []
       
    def __str__(self):
        return "<{}반>  {}명".format(self.no, self.count_student())
    
    def __lt__(self,other):
        return self.no < other.no
    
    def __eq__(self,other):
        return self.no == other.no
    
    def count_student(self):
        return len(self.student_list)
        #ban1.count_student()