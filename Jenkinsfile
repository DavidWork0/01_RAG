pipeline {
    agent any

    stages {
        stage('Setup Environment') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'python3 -m venv .venv'
                        sh '. .venv/bin/activate && pip install --upgrade pip'
                        // Install torch with CUDA support
                        sh '. .venv/bin/activate && pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128'
                        // Install other requirements
                        sh '. .venv/bin/activate && pip install -r requirements_project_without_torch.txt'
                    } else {
                        bat 'python -m venv .venv'
                        bat '.venv\\Scripts\\activate.bat && pip install --upgrade pip'
                        // Install torch with CUDA support
                        bat '.venv\\Scripts\\activate.bat && pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128'
                        // Install other requirements
                        bat '.venv\\Scripts\\activate.bat && pip install -r requirements_project_without_torch.txt'
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    if (isUnix()) {
                        // Run standard unit tests
                        sh '. .venv/bin/activate && python tests/test_full.py'
                        sh '. .venv/bin/activate && python tests/test_reranking.py'
                        
                        // Run similarity filter test script
                        sh '. .venv/bin/activate && python tests/test_similarity_filter.py'
                        
                        // Run inference test (quick mode)
                        sh '. .venv/bin/activate && python tests/test_inference.py --model Qwen3-8B-Q5_K_M --mode quick'
                    } else {
                        // Run standard unit tests
                        bat '.venv\\Scripts\\activate.bat && python tests\\test_full.py'
                        bat '.venv\\Scripts\\activate.bat && python tests\\test_reranking.py'
                        
                        // Run similarity filter test script
                        bat '.venv\\Scripts\\activate.bat && python tests\\test_similarity_filter.py'
                        
                        // Run inference test (quick mode)
                        bat '.venv\\Scripts\\activate.bat && python tests\\test_inference.py --model Qwen3-8B-Q5_K_M --mode quick'
                    }
                }
            }
        }
    }
}
