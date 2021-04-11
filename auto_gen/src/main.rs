use fake::{Dummy, Fake, Faker};
use serde::Serialize;
use std::fs::File;
use std::error::Error;
//use rand::Rng;
use fake::faker::company::en::*;
use fake::faker::name::en::*;
use fake::faker::boolean::en::*;
use fake::faker::option::raw::*;

#[derive(Dummy, Serialize)]
pub struct Foo {
    #[dummy(faker = "Opt(0..2000, 93)")]
    order_id: Option<u64>,
    #[dummy(faker = "Opt(0..2000, 81)")]
    make_id: Option<u64>,
    #[dummy(faker = "Opt(0..2000, 98)")]
    time_id: Option<u64>,
    #[dummy(faker = "Opt(0..2000, 90)")]
    this_id: Option<u64>,
    #[dummy(faker = "Opt(0.0..2000.0, 78)")]
    my_float: Option<f64>,
    #[dummy(faker = "0.0..2000.0")]
    not_null: f64,
    #[dummy(faker = "0..2000")]
    not_null_int: u64,
    #[dummy(faker = "Name()")]
    not_null_str: String,
    #[dummy(faker = "Opt(CompanyName(), 97)")]
    customer: Option<String>,
    #[dummy(faker = "Opt(Boolean(10), 60)")]
    paid: Option<bool>,
}


fn main() -> Result<(), Box<dyn Error>>{
//    let mut rng = rand::thread_rng();
    let file = File::create("data.csv")?;
    let mut wtr = csv::Writer::from_writer(file);
    let mut i = 0;
//    let rows = rng.gen_range(10000..100000);
    let rows = 10000;
    loop {
        wtr.serialize(Faker.fake::<Foo>())?;
        if i == rows {
            break;
        }
        i += 1;
    }
    wtr.flush()?;
    Ok(())
}

/*use fake::{Dummy, Fake, Faker};
use serde::Serialize;
use std::fs::File;
use std::error::Error;
use rand::Rng;
use fake::faker::company::en::*;
use fake::faker::name::en::*;
use fake::faker::boolean::en::*;

#[derive(Dummy, Serialize)]
pub struct Foo {
    #[dummy(faker = "0..2000")]
    order_id: Option<u64>,
    #[dummy(faker = "0..2000")]
    make_id: Option<u64>,
    #[dummy(faker = "0..2000")]
    time_id: Option<u64>,
    #[dummy(faker = "0..2000")]
    this_id: Option<u64>,
    #[dummy(faker = "0.0..2000.0")]
    my_float: Option<f64>,
    #[dummy(faker = "0.0..2000.0")]
    not_null: f64,
    #[dummy(faker = "0..2000")]
    not_null_int: u64,
    #[dummy(faker = "Name()")]
    not_null_str: String,
    #[dummy(faker = "CompanyName()")]
    customer: Option<String>,
    #[dummy(faker = "Boolean(10)")]
    paid: Option<bool>,
}


fn main() -> Result<(), Box<dyn Error>>{
    let mut rng = rand::thread_rng();
    let file = File::create("data.csv")?;
    let mut wtr = csv::Writer::from_writer(file);
    let mut i = 0;
    let rows = rng.gen_range(10000..100000);
    loop {
        let f: Foo = Faker.fake_with_rng(rng)
        wtr.serialize(f)?;
        if i == rows {
            break;
        }
        i += 1;
    }
    wtr.flush()?;
    Ok(())
}*/
