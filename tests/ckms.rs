mod integration {
    mod ckms {
        extern crate quantiles;

        use self::quantiles::ckms::CKMS;
        use std::fs::File;
        use std::io::Read;
        use std::path::PathBuf;
        use std::str::FromStr;

        #[test]
        fn test_run_afl_examples() {
            let mut resource = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            resource.push("resources/afl_crashes_20161215.txt");

            let mut f = File::open(resource).expect("could not open resource file");
            let mut buffer = String::new();
            f.read_to_string(&mut buffer)
                .expect("could not read resource file");

            for s in buffer.lines() {
                let pyld: Vec<f64> = s.split_whitespace()
                    .map(|f| f64::from_str(f))
                    .filter(|f| f.is_ok())
                    .map(|f| f.unwrap())
                    .collect();

                if pyld.len() >= 2 {
                    let mut ckms = CKMS::new(pyld[0]);

                    for f in &pyld[1..] {
                        ckms.insert(*f)
                    }
                }
            }
        }
    }
}
