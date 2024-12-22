import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Main {

	static Scanner scanner = new Scanner(System.in);
    public static void main(String[] args) {
        Map<Integer, Runnable> menuActions = new HashMap<>();
        
        menuActions.put(1, () -> trainAndTestWithMomentum());
        menuActions.put(2, () -> {
			try {
				trainAndTestWithoutMomentum();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
        menuActions.put(3, () -> {
			try {
				trainWithEpochDisplay();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
        menuActions.put(4, () -> trainAndSingleTestWithMomentum());
        menuActions.put(5, () -> {
			try {
				kFoldTest();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});

        while (true) {
            System.out.println("\n--- Menü ---");
            System.out.println("1- Ağı Eğit ve Test Et (Momentumlu)");
            System.out.println("2- Ağı Eğit ve Test Et (Momentumsuz)");
            System.out.println("3- Ağı Eğit Epoch Göster");
            System.out.println("4- Ağı Eğit ve Tekli Test (Momentumlu)");
            System.out.println("5- K-Fold Test");
            System.out.println("0- Çıkış");
            System.out.print("Bir seçenek giriniz: ");

            int choice = scanner.nextInt();
            if (choice == 0) {
                System.out.println("Programdan çıkıldı");
                break;
            }

            Runnable action = menuActions.get(choice);
            if (action != null) {
                action.run();
            } else {
                System.out.println("Geçersiz seçim lütfen tekrar deneyin.");
            }
        }
    }

    private static void trainAndTestWithMomentum() {
        System.out.println("Ağın momentumlu eğitilip test edilmesi işlemi başlıyor...");
        BPMomentum bpm = new BPMomentum(0.7,700);
        bpm.train();
    }

    private static void trainAndTestWithoutMomentum() throws FileNotFoundException {
    	System.out.println("Ağın momentumsuz eğitilip test edilmesi süreci başlıyor...");
        BP bp = new BP(0.0001,700,0.01);
        bp.train();
    }

    private static void trainWithEpochDisplay() throws FileNotFoundException {
    	System.out.println("Ağın momentumsuz eğitilip epochlara bölünmüş olarak test edilmesi işlemi başlıyor...");
        BPshowepoch  bps = new BPshowepoch(0.0001,700,0.01);
        bps.egitEpochGoster();
    }

    private static void trainAndSingleTestWithMomentum() {
    	System.out.println("Ağın momentumlı eğitilip kullanıcı girdilerine göre sonuç verilmesi işlemi başlıyor...");

    	double momentum = 0.0;
    	double ogrenme = 0.0;
    	double error = 0.0;
    	int epoch = 0;
	
    	while (true) {
    	    System.out.println("Momentum giriniz [0-1] arasında olmalıdır:");
    	    if (scanner.hasNextDouble()) {
    	        momentum = scanner.nextDouble();
    	        if (momentum >= 0.0 && momentum <= 1.0) {
    	            break;
    	        } else {
    	            System.out.println("Hata: Momentum 0 ile 1 arasında olmalıdır!");
    	        }
    	    } else {
    	        System.out.println("Hata: Geçersiz giriş. Lütfen bir ondalıklı sayı giriniz!");
    	        scanner.next(); 
    	    }
    	}

    	
    	while (true) {
    	    System.out.println("Öğrenme katsayısını giriniz (pozitif sayı):");
    	    if (scanner.hasNextDouble()) {
    	        ogrenme = scanner.nextDouble();
    	        if (ogrenme > 0.0) {
    	            break;
    	        } else {
    	            System.out.println("Hata: Öğrenme katsayısı pozitif bir sayı olmalıdır!");
    	        }
    	    } else {
    	        System.out.println("Hata: Geçersiz giriş. Lütfen bir ondalıklı sayı giriniz!");
    	        scanner.next(); 
    	    }
    	}

    	
    	while (true) {
    	    System.out.println("Programın duracağı hata miktarını belirtiniz (pozitif sayı):");
    	    if (scanner.hasNextDouble()) {
    	        error = scanner.nextDouble();
    	        if (error > 0.0) {
    	            break;
    	        } else {
    	            System.out.println("Hata: Hata miktarı pozitif bir sayı olmalıdır!");
    	        }
    	    } else {
    	        System.out.println("Hata: Geçersiz giriş. Lütfen bir ondalıklı sayı giriniz!");
    	        scanner.next(); 
    	    }
    	}

    	
    	while (true) {
    	    System.out.println("Epoch sayısını giriniz (pozitif tam sayı):");
    	    if (scanner.hasNextInt()) {
    	        epoch = scanner.nextInt();
    	        if (epoch > 0) {
    	            break;
    	        } else {
    	            System.out.println("Hata: Epoch sayısı pozitif bir sayı olmalıdır!");
    	        }
    	    } else {
    	        System.out.println("Hata: Geçersiz giriş. Lütfen bir tam sayı giriniz!");
    	        scanner.next();
    	    }
    	}
    	BPMomentum bpMomentum = new BPMomentum(momentum, epoch);
    	bpMomentum.train();
    	Scanner in = new Scanner(System.in);
    	System.out.println("Tohum Miktarını giriniz(1KG - 40KG):");
    	double tohumMiktari = in.nextInt();
    	System.out.println("Su Miktarını Giriniz(12,5TON - 800TON):");
    	double suMiktari = in.nextInt();
    	System.out.println("Güneş Işığı Alınan Saati Giriniz(1.5-24):");
    	double gunesIsigi = in.nextInt();
    	double pred = bpMomentum.testValue(tohumMiktari, suMiktari, gunesIsigi);
    	System.out.println("Elde edilecek buğday miktarı(TON): "+ pred/1000);
    }

    private static void kFoldTest() throws FileNotFoundException {
        System.out.println("K-Fold test secildi");
        System.out.println("K değerini giriniz: ");
        int k = scanner.nextInt();
        Kfold kfold = new Kfold(k);
        kfold.performKFoldValidation();
    }
}
