import mpi.MPI;

import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;

public class Main {

    public static void main(String[] args) throws IOException {
        MPI.Init(args);
        int me = MPI.COMM_WORLD.Rank();
        if (me == 0) {
            // master process
            Matrix matrix = Matrix.fromFile();
            searchMaster(matrix);
        } else {
            // worker process
            searchWorker();
        }
        MPI.Finalize();
    }

    private static void searchMaster(Matrix root) {
        int numberOfWorkersPlusMaster = MPI.COMM_WORLD.Size();
        int numberOfWorkers = numberOfWorkersPlusMaster - 1;
        int originalMatrixBound = root.getManhattan();
        boolean found = false;
        long time = System.currentTimeMillis();

        Queue<Matrix> matrixMoveQueue = new LinkedList<>();
        matrixMoveQueue.add(root);
        // generate as many possible moves as the number of workers
        while (matrixMoveQueue.size() + matrixMoveQueue.peek().generateMoves().size() - 1 <= numberOfWorkers) {
            Matrix curr = matrixMoveQueue.poll();
            for (Matrix neighbour : curr.generateMoves()) {
                matrixMoveQueue.add(neighbour);
            }
        }

        while (!found) {
            Queue<Matrix> currentMatricesQueue = new LinkedList<>(matrixMoveQueue);
            for (int i = 0; i < matrixMoveQueue.size(); i++) {
                Matrix currentMatrix = currentMatricesQueue.poll();
                MPI.COMM_WORLD.Send(new boolean[]{false}, 0, 1, MPI.BOOLEAN, i + 1, 0);
                MPI.COMM_WORLD.Send(new Object[]{currentMatrix}, 0, 1, MPI.OBJECT, i + 1, 0);
                MPI.COMM_WORLD.Send(new int[]{originalMatrixBound}, 0, 1, MPI.INT, i + 1, 0);
            }

            Object[] workerResultingPairs = new Object[numberOfWorkersPlusMaster + 5];
            for (int i = 1; i <= matrixMoveQueue.size(); i++) {
                MPI.COMM_WORLD.Recv(workerResultingPairs, i - 1, 1, MPI.OBJECT, i, 0);
            }

            // check if any node found a solution
            int newMinBound = Integer.MAX_VALUE;
            for (int i = 0; i < matrixMoveQueue.size(); i++) {
                Pair<Integer, Matrix> workerSolutionPair = (Pair<Integer, Matrix>) workerResultingPairs[i];
                if (workerSolutionPair.getFirst() == -1) {
                    // found solution
                    System.out.println("Solution found in " + workerSolutionPair.getSecond().getNumOfSteps() + " steps");
                    System.out.println("Solution is: ");
                    System.out.println(workerSolutionPair.getSecond());
                    System.out.println("Execution time: " + (System.currentTimeMillis() - time) + "ms");
                    found = true;
                    break;
                } else if (workerSolutionPair.getFirst() < newMinBound) {
                    newMinBound = workerSolutionPair.getFirst();
                }
            }
            if(!found){
                System.out.println("Depth " + newMinBound + " reached in " + (System.currentTimeMillis() - time) + "ms");
                originalMatrixBound = newMinBound;
            }
        }

        for (int i = 1; i < numberOfWorkersPlusMaster; i++) {
            // shut down workers when solution was found
            Matrix curr = matrixMoveQueue.poll();
            MPI.COMM_WORLD.Send(new boolean[]{true}, 0, 1, MPI.BOOLEAN, i, 0);
            MPI.COMM_WORLD.Send(new Object[]{curr}, 0, 1, MPI.OBJECT, i, 0);
            MPI.COMM_WORLD.Send(new int[]{originalMatrixBound}, 0, 1, MPI.INT, i, 0);
        }
    }

    private static void searchWorker() {
        while (true) {
            Object[] matrix = new Object[1];
            int[] bound = new int[1];
            boolean[] end = new boolean[1];
            MPI.COMM_WORLD.Recv(end, 0, 1, MPI.BOOLEAN, 0, 0);
            MPI.COMM_WORLD.Recv(matrix, 0, 1, MPI.OBJECT, 0, 0);
            MPI.COMM_WORLD.Recv(bound, 0, 1, MPI.INT, 0, 0);
            if (end[0]) { // shut down when solution was found
                return;
            }
            int originalMatrixBound = bound[0];
            Matrix current = (Matrix) matrix[0];
            Pair<Integer, Matrix> result = search(current, current.getNumOfSteps(), originalMatrixBound);
            MPI.COMM_WORLD.Send(new Object[]{result}, 0, 1, MPI.OBJECT, 0, 0);
        }
    }

    public static Pair<Integer, Matrix> search(Matrix current, int numSteps, int bound) {
        int estimation = numSteps + current.getManhattan();
        if (estimation > bound) {
            return new Pair<>(estimation, current);
        }
        if (estimation > 80) {
            return new Pair<>(estimation, current);
        }
        if (current.getManhattan() == 0) {
            return new Pair<>(-1, current);
        }
        int min = Integer.MAX_VALUE;
        Matrix solution = null;
        for (Matrix next : current.generateMoves()) {
            Pair<Integer, Matrix> result = search(next, numSteps + 1, bound);
            int t = result.getFirst();
            if (t == -1) {
                return new Pair<>(-1, result.getSecond());
            }
            if (t < min) {
                min = t;
                solution = result.getSecond();
            }
        }
        return new Pair<>(min, solution);
    }

}