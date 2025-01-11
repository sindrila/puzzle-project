import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class IDAStar {
    private static final int NR_THREADS = 5;
    private static final int NR_TASKS = 5;
    private static ExecutorService executorService;

    public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
        Matrix initialState = Matrix.fromFile();

        executorService = Executors.newFixedThreadPool(NR_THREADS);

        Matrix solution = solve(initialState);
        System.out.println(solution);
        executorService.shutdown();
        executorService.awaitTermination(1000000, TimeUnit.SECONDS);
    }


    public static Matrix solve(Matrix root) throws ExecutionException, InterruptedException {
        long time = System.currentTimeMillis();
        // min bound represents the heuristic
        // how far the current state is from the solution
        int minBound = root.getManhattan();
        int dist;
        while (true) {
            Pair<Integer, Matrix> solution = searchParallel(root, 0, minBound, NR_TASKS);
            dist = solution.getFirst();
            if (dist == -1) {
                System.out.println("Solution found in " + solution.getSecond().getNumOfSteps() + " steps");
                System.out.println("Execution time: " + (System.currentTimeMillis() - time) + "ms");
                return solution.getSecond();
            } else {
                System.out.println("Depth " + dist + " reached in " + (System.currentTimeMillis() - time) + "ms");
            }
            minBound = dist;
        }
    }


    public static Pair<Integer, Matrix> searchParallel(Matrix currentMatrix, int numSteps, int originalMatrixBound, int searchDepth) throws ExecutionException, InterruptedException {
        if (searchDepth <= 1) {
            return search(currentMatrix, numSteps, originalMatrixBound);
        }

        int currentManhattanSum = currentMatrix.getManhattan();
        int estimation = numSteps + currentManhattanSum;
        if (estimation > originalMatrixBound || estimation > 80) {
            return new Pair<>(estimation, currentMatrix);
        }
        if (currentManhattanSum == 0) {
            return new Pair<>(-1, currentMatrix);
        }
        int minimumSteps = Integer.MAX_VALUE;
        List<Matrix> moves = currentMatrix.generateMoves();
        List<Future<Pair<Integer, Matrix>>> futures = new ArrayList<>();
        for (Matrix possibleMove : moves) {
            Future<Pair<Integer, Matrix>> f = executorService.submit(() -> searchParallel(possibleMove, numSteps + 1, originalMatrixBound, searchDepth / moves.size()));
            futures.add(f);
        }
        for (Future<Pair<Integer, Matrix>> f : futures) {
            Pair<Integer, Matrix> result = f.get();
            int t = result.getFirst();
            if (t == -1) {
                return new Pair<>(-1, result.getSecond());
            }
            if (t < minimumSteps) {
                minimumSteps = t;
            }
        }
        return new Pair<>(minimumSteps, currentMatrix);
    }

    public static Pair<Integer, Matrix> search(Matrix current, int numSteps, int originalMatrixBound) {
        int currentManhattanSum = current.getManhattan();
        int estimation = numSteps + currentManhattanSum;
        if (estimation > originalMatrixBound || estimation > 80) {
            return new Pair<>(estimation, current);
        }
        if (currentManhattanSum == 0) {
            return new Pair<>(-1, current);
        }
        int min = Integer.MAX_VALUE;
        Matrix solution = null;
        for (Matrix next : current.generateMoves()) {
            Pair<Integer, Matrix> result = search(next, numSteps + 1, originalMatrixBound);
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