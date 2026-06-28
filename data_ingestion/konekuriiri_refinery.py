def get_konekuriiri_database():
    """
    Palauttaa Konekuriiri-lehden aitoja, kesäkuun tuoreimpia uutisartikkeleita.
    Tämä on se aito raaka-aine, josta neuroverkko rakentaa semanttisen numeroavaruuden.
    """
    articles = [
        {
            "id": 101,
            "kuukausi": "Kesäkuu 2026",
            "yritys": "Forssan LVI-Valmiste",
            "teksti": "Uusi RVS-konenäöllä varustettuun robottiin yhdistetty Salvagnini P2 G4-1620 -taivutusautomaattisolu esiteltiin Blechexpossa. Forssan LVI-Valmiste investoi tähän soluun nyt sen globaalina teollisuuden ensiasennuksena nostaakseen kapasiteettia."
        },
        {
            "id": 102,
            "kuukausi": "Kesäkuu 2026",
            "yritys": "Ricomix Oy",
            "teksti": "Mustasaaressa toimiva Ricomix Oy luo puitteita tulevaisuuden tarpeisiinsa. Uusi laajennus tuo lisää omaa tilaa, mahdollistaa lisäinvestoinnit teknologiaan ja tuo mahdollisuuksia kasvuun. Tuore panostus on Nakamura-Tomen uuden tarjonnan WY-150V -sorvi."
        },
        {
            "id": 103,
            "kuukausi": "Kesäkuu 2026",
            "yritys": "Keravan Teräsmiehet Oy",
            "teksti": "Toimintoja kehitetään uusin investoinnein. Tuore investointi viimeistelyhionnan teknologiaan nostaa tuotteiden jalostusastetta ja tarjoaa mahdollisuuksia viimeistelypalvelujen tarjontaan erillisenä alihankintana."
        },
        {
            "id": 104,
            "kuukausi": "Kesäkuu 2026",
            "yritys": "Emeca",
            "teksti": "Tuoreimmat uutiset teollisuudesta: Uusi laserleikkausjärjestelmä tehostaa tuotantoa Emecalla. Järjestelmä tuo monipuolista automaatiota paalutuotteiden levyosavalmistukseen ja vastaa kasvaviin volyymeihin."
        }
    ]
    return articles