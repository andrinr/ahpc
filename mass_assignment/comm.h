#ifndef COMM_H_INCLUDED
#define COMM_H_INCLUDED

class Communicator {
public:
    int np, rank;
    Communicator();
    ~Communicator();

private:
    int errs;
    int provided, flag, claimed;
};

#endif // COMM_H_INCLUDED