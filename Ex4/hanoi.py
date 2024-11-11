import itertools
import sys


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name,
                       'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    domain_file.write("Propositions:\n")

    # Write all possible valid combinations
    for d in disks:
        for p in pegs:
            domain_file.write(f"{d}_restsOn_{p} ")
    for d1, d2 in itertools.combinations(disks, r=2):
        domain_file.write(f"{d1}_restsOn_{d2} ")
    for d in disks:
        domain_file.write(f"free_{d} ")
    for i,p in enumerate(pegs):
        if i < len(pegs)-1:
            domain_file.write(f"free_{p} ")
    domain_file.write(f"free_{pegs[-1]} ")

    domain_file.write("\nActions:\n")


    for d in disks:
        for p1 in pegs:
            for p2 in pegs:
                if p1 != p2:
                    domain_file.write(
                        f"Name: Levitate_{d}_from_{p1}_to_{p2}\n")
                    domain_file.write(
                        f"pre: free_{d} {d}_restsOn_{p1} free_{p2}\n")
                    domain_file.write(
                        f"add: {d}_restsOn_{p2} free_{p1}\n")
                    domain_file.write(
                        f"delete: {d}_restsOn_{p1} free_{p2}\n")

    # Generate levitation spells onto larger disks
    for i, d1 in enumerate(disks):
        for j, d2 in enumerate(disks[i + 1:], start=i + 1):
            for p in pegs:
                domain_file.write(
                    f"Name: Levitate_{d1}_from_{p}_to_{d2}\n")
                domain_file.write(
                    f"pre: free_{d1} {d1}_restsOn_{p} free_{d2} \n")
                domain_file.write(f"add: {d1}_restsOn_{d2} free_{p}\n")
                domain_file.write(
                    f"delete: free_{d2} {d1}_restsOn_{p}\n")

                domain_file.write(
                    f"Name: Levitate_{d1}_from_{d2}_to_{p}\n")
                domain_file.write(
                    f"pre: free_{d1} {d1}_restsOn_{d2} free_{p} \n")
                domain_file.write(f"add: {d1}_restsOn_{p} free_{d2}\n")
                domain_file.write(
                    f"delete: free_{p} {d1}_restsOn_{d2}\n")

    for i, d in enumerate(disks):
        for j, d2 in enumerate(disks[i + 1:], start=i + 1):
            for d1 in disks[i + 2:]:
                if d2 != d1:
                    domain_file.write(
                        f"Name: Levitate_{d}_from_{d1}_to_{d2}\n")
                    domain_file.write(
                        f"pre: free_{d} {d}_restsOn_{d1} free_{d2} \n")
                    domain_file.write(f"add: {d}_restsOn_{d2} free_{d1}\n")
                    domain_file.write(
                        f"delete: free_{d2} {d}_restsOn_{d1}\n")

                    domain_file.write(
                        f"Name: Levitate_{d}_from_{d2}_to_{d1}\n")
                    domain_file.write(
                        f"pre: free_{d} {d}_restsOn_{d2} free_{d1} \n")
                    domain_file.write(f"add: {d}_restsOn_{d1} free_{d2}\n")
                    domain_file.write(
                        f"delete: free_{d1} {d}_restsOn_{d2}\n")
    domain_file.close()



def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"
    problem_file.write("Initial state: ")

    problem_file.write(f"{disks[-1]}_restsOn_{pegs[0]} ")
    for i in range(n_ - 1):
        problem_file.write(f"{disks[i]}_restsOn_{disks[i + 1]} ")
    problem_file.write(f"free_{disks[0]} ")
    for p in pegs[1:]:
        problem_file.write(f"free_{p} ")

    problem_file.write("\nGoal state: ")

    problem_file.write(f"{disks[-1]}_restsOn_{pegs[-1]} ")
    for i in range(n_ - 1):
        problem_file.write(f"{disks[i]}_restsOn_{disks[i + 1]} ")


    problem_file.close()



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
