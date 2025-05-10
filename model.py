def CNN():
  network = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(num_ch-1,1), padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.Conv2d(64, 64, 3, padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(),

                          nn.MaxPool2d(2, 2), # BSx64x16x16

                          nn.Conv2d(64, 128, 3, padding=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.Conv2d(128, 128, 3, padding=1),
                          nn.BatchNorm2d(128),
                          nn.ReLU(),

                          nn.MaxPool2d(2,2), # 8x8

                          nn.Conv2d(128, 256, 3, padding=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),

                          nn.Conv2d(256, 256, 3, padding=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU(),
                          # BSx256x8x8 -> BSx256x1x1
                          nn.AdaptiveAvgPool2d(output_size=(1, 1)), # BS1x1

                          nn.Flatten(), # BSx256
                          nn.Linear(256, 9)
                      )

  return network